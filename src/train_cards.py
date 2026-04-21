from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def _bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()


    p.add_argument("--data-root", default=os.environ.get("SM_CHANNEL_TRAIN", ""))
    p.add_argument(
        "--unpack-dir",
        default=os.environ.get("SM_CHANNEL_UNPACK", "/opt/ml/input/data/unpacked"),
        help="Where to unzip cards.zip when the channel contains a zip instead of ImageFolder tree.",
    )
    p.add_argument("--train-dir", default="")
    p.add_argument("--valid-dir", default="")
    p.add_argument("--test-dir", default="")

    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", str(Path("outputs") / "model")))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", str(Path("outputs"))))

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)

    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--no-plots", action="store_true", help="skip matplotlib figures (headless / CI)")
    args = p.parse_args()


    def _hp(name: str, cast, fallback):
        v = os.environ.get(f"SM_HP_{name.upper().replace('-', '_')}")
        if v is None or v == "":
            return fallback
        return cast(v)

    args.seed = _hp("seed", int, args.seed)
    args.img_size = _hp("img_size", int, args.img_size)
    args.batch_size = _hp("batch_size", int, args.batch_size)
    args.num_workers = _hp("num_workers", int, args.num_workers)
    args.max_epochs = _hp("max_epochs", int, args.max_epochs)
    args.patience = _hp("patience", int, args.patience)
    args.min_delta = _hp("min_delta", float, args.min_delta)
    args.lr_head = _hp("lr_head", float, args.lr_head)
    args.lr_backbone = _hp("lr_backbone", float, args.lr_backbone)
    args.weight_decay = _hp("weight_decay", float, args.weight_decay)

    if os.environ.get("SM_HP_NO_PLOTS"):
        args.no_plots = _bool(os.environ["SM_HP_NO_PLOTS"])

    return args


def _find_zip(root: Path) -> Path | None:
    if root.is_file() and root.suffix.lower() == ".zip":
        return root
    if root.is_dir():
        zips = sorted(root.glob("*.zip"))
        if len(zips) == 1:
            return zips[0]
        for name in ("cards.zip", "dataset.zip"):
            cand = root / name
            if cand.is_file():
                return cand
    return None


def _ensure_imagefolder_root(data_root: Path, unpack_dir: Path) -> Path:

    if (data_root / "train").is_dir() and (data_root / "valid").is_dir() and (data_root / "test").is_dir():
        return data_root

    zpath = _find_zip(data_root)
    if zpath is None:
        raise SystemExit(
            f"Could not find train/valid/test under {data_root} and no .zip to unpack. "
            f"Upload either the folder tree or a single cards.zip."
        )

    unpack_root = Path(unpack_dir)
    marker = unpack_root / ".cards_unpacked_marker.txt"
    if marker.is_file() and (unpack_root / "train").is_dir():
        return unpack_root

    unpack_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmp_path)

        train_dir = None
        for p in tmp_path.rglob("train"):
            if p.is_dir() and (p.parent / "valid").is_dir() and (p.parent / "test").is_dir():
                train_dir = p
                break
        if train_dir is None:
            raise SystemExit(f"Unpacked {zpath} but could not find train/valid/test folders inside.")

        parent = train_dir.parent
        if unpack_root.exists():
            shutil.rmtree(unpack_root)
        shutil.copytree(parent, unpack_root)
        marker.write_text(f"unpacked_from={zpath}\n", encoding="utf-8")

    return unpack_root


def resolve_data_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.train_dir and args.valid_dir and args.test_dir:
        return Path(args.train_dir), Path(args.valid_dir), Path(args.test_dir)
    root = Path(args.data_root)
    if not str(root):
        raise SystemExit("Provide --data-root (folder containing train/ valid/ test/) or explicit --train-dir/--valid-dir/--test-dir.")
    root = _ensure_imagefolder_root(root, Path(args.unpack_dir))
    return root / "train", root / "valid", root / "test"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf, mean, std


def run_epoch(model, loader, criterion, optim, device: torch.device, train: bool):
    model.train(train)
    tot_loss = 0.0
    tot_correct = 0
    tot = 0
    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
            tot_loss += loss.item() * x.size(0)
            tot_correct += int((logits.argmax(1) == y).sum().item())
            tot += x.size(0)
    return tot_loss / max(tot, 1), tot_correct / max(tot, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir, valid_dir, test_dir = resolve_data_dirs(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, eval_tf, mean, std = build_transforms(args.img_size)

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    valid_ds = datasets.ImageFolder(str(valid_dir), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tf)

    classes = train_ds.classes
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))
    model = model.to(device)

    head_params = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    optim = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.max_epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None
    epochs_no_improve = 0
    stopped_early = False

    t0 = time.time()
    for ep in range(1, args.max_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optim, device, train=True)
        vl_loss, vl_acc = run_epoch(model, valid_loader, criterion, optim, device, train=False)
        scheduler.step()

        for k, v in zip(history.keys(), (tr_loss, tr_acc, vl_loss, vl_acc)):
            history[k].append(v)

        if best_state is None:
            best_val = vl_acc
            best_state = copy.deepcopy(model.state_dict())
            marker = "  *init*"
            epochs_no_improve = 0
        elif vl_acc > best_val + args.min_delta:
            best_val = vl_acc
            best_state = copy.deepcopy(model.state_dict())
            marker = "  *best*"
            epochs_no_improve = 0
        else:
            marker = ""
            epochs_no_improve += 1

        print(
            f"ep {ep:2d}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
            f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}{marker}"
        )

        if epochs_no_improve >= args.patience:
            stopped_early = True
            print(
                f"\nearly stopping: val_acc did not beat best+{args.min_delta} "
                f"for {args.patience} epochs (best val_acc={best_val:.4f})"
            )
            break

    train_time = time.time() - t0
    epochs_ran = len(history["train_loss"])
    print(f"\ntrained {epochs_ran} epoch(s) in {train_time/60:.1f} min")
    model.load_state_dict(best_state) 


    all_logits, all_y = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in test_loader:
            all_logits.append(model(x.to(device)).cpu())
            all_y.append(y)
    all_logits = torch.cat(all_logits)
    all_y = torch.cat(all_y)
    preds = all_logits.argmax(1)
    test_acc = float((preds == all_y).float().mean().item())
    print(f"test top-1 accuracy: {test_acc:.4f}")

    print(f"METRIC test_top1_accuracy = {test_acc:.6f}")
    print(f"METRIC best_val_accuracy = {float(best_val):.6f}")
    print(f"METRIC epochs_ran = {float(epochs_ran)}")

    report = classification_report(
        all_y.numpy(), preds.numpy(), target_names=classes, digits=4, output_dict=True
    )
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        ep_x = list(range(1, len(history["train_loss"]) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(ep_x, history["train_loss"], marker="o", label="train")
        ax1.plot(ep_x, history["val_loss"], marker="s", label="valid")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.set_title("Loss")
        ax1.legend()

        ax2.plot(ep_x, history["train_acc"], marker="o", label="train")
        ax2.plot(ep_x, history["val_acc"], marker="s", label="valid")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("accuracy")
        ax2.set_ylim(0, 1.01)
        ax2.set_title("Accuracy")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close(fig)

        cm = confusion_matrix(all_y.numpy(), preds.numpy())
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm,
            cmap="Blues",
            cbar=False,
            xticklabels=classes,
            yticklabels=classes,
            square=True,
            linewidths=0.1,
            ax=ax,
        )
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Confusion matrix (test)  —  top-1 acc = {test_acc:.3f}")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    torch.save(model.state_dict(), model_dir / "model.pt")
    with open(model_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    meta = {
        "model": "efficientnet_b0",
        "num_classes": len(classes),
        "img_size": args.img_size,
        "max_epochs": args.max_epochs,
        "epochs_ran": epochs_ran,
        "early_stopped": stopped_early,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "best_val_acc": best_val,
        "batch_size": args.batch_size,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "train_seconds": train_time,
        "test_top1_acc": test_acc,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "imagenet_mean": mean,
        "imagenet_std": std,
        "train_dir": str(train_dir),
        "valid_dir": str(valid_dir),
        "test_dir": str(test_dir),
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    tar_path = model_dir / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_dir / "model.pt", arcname="model.pt")
        tar.add(model_dir / "classes.json", arcname="classes.json")
        tar.add(model_dir / "metadata.json", arcname="metadata.json")

    print(f"wrote {tar_path} ({tar_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
