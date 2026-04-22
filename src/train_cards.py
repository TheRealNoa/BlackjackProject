from __future__ import annotations

import argparse
import csv
import copy
import datetime
import io
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


def _no_plots_cli_value(s: str) -> bool:
    sl = str(s).lower().strip()
    if sl in ("0", "false", "no", "n", ""):
        return False
    return True


def _extract_sm_context() -> dict:
    ctx = {
        "training_job_name": os.environ.get("TRAINING_JOB_NAME", ""),
        "current_host": os.environ.get("SM_CURRENT_HOST", ""),
        "hosts": os.environ.get("SM_HOSTS", ""),
        "instance_type": os.environ.get("SM_CURRENT_INSTANCE_TYPE", ""),
        "num_gpus": os.environ.get("SM_NUM_GPUS", ""),
        "num_cpus": os.environ.get("SM_NUM_CPUS", ""),
    }
    raw = os.environ.get("SM_TRAINING_ENV")
    if raw:
        try:
            env_obj = json.loads(raw)
            ctx["training_job_name"] = ctx["training_job_name"] or env_obj.get("job_name", "")
            ctx["current_host"] = ctx["current_host"] or env_obj.get("current_host", "")
            if not ctx["hosts"] and "hosts" in env_obj:
                ctx["hosts"] = env_obj["hosts"]
            if not ctx["instance_type"] and "current_instance_type" in env_obj:
                ctx["instance_type"] = env_obj["current_instance_type"]
            if not ctx["num_gpus"] and "num_gpus" in env_obj:
                ctx["num_gpus"] = env_obj["num_gpus"]
            if not ctx["num_cpus"] and "num_cpus" in env_obj:
                ctx["num_cpus"] = env_obj["num_cpus"]
        except json.JSONDecodeError:
            pass
    return ctx


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    no_scheme = uri[5:]
    if "/" not in no_scheme:
        raise ValueError(f"S3 URI must include key path, got: {uri}")
    bucket, key = no_scheme.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI, got: {uri}")
    return bucket, key


def _append_metrics_csv_to_s3(s3_uri: str, row: dict) -> None:
    if not s3_uri:
        return

    try:
        import boto3
        from botocore.exceptions import ClientError
    except Exception as e:
        print(f"WARN could not import boto3/botocore for metrics CSV upload: {e}")
        return

    try:
        bucket, key = _parse_s3_uri(s3_uri)
    except ValueError as e:
        print(f"WARN metrics CSV upload skipped: {e}")
        return

    s3 = boto3.client("s3")
    existing_rows = []
    fieldnames = list(row.keys())
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(body))
        existing_rows = list(reader)
        if reader.fieldnames:
            merged = list(reader.fieldnames)
            for fn in fieldnames:
                if fn not in merged:
                    merged.append(fn)
            fieldnames = merged
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("NoSuchKey", "404"):
            print(f"WARN failed to read existing metrics CSV {s3_uri}: {e}")
            return

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in existing_rows:
        writer.writerow({k: r.get(k, "") for k in fieldnames})
    writer.writerow({k: row.get(k, "") for k in fieldnames})

    try:
        s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"), ContentType="text/csv")
        print(f"wrote run-metrics CSV row to {s3_uri}")
    except Exception as e:
        print(f"WARN failed to upload metrics CSV to {s3_uri}: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SageMaker passes hyperparameters as underscore flags (--batch_size) "
    )

    p.add_argument("--data-root", "--data_root", default=os.environ.get("SM_CHANNEL_TRAIN", ""))
    p.add_argument(
        "--unpack-dir",
        "--unpack_dir",
        default=os.environ.get("SM_CHANNEL_UNPACK", "/opt/ml/input/data/unpacked"),
        help="Where to unzip cards.zip when the channel contains a zip instead of ImageFolder tree.",
    )
    p.add_argument("--train-dir", "--train_dir", default="")
    p.add_argument("--valid-dir", "--valid_dir", default="")
    p.add_argument("--test-dir", "--test_dir", default="")

    p.add_argument(
        "--model-dir",
        "--model_dir",
        default=os.environ.get("SM_MODEL_DIR", str(Path("outputs") / "model")),
    )
    p.add_argument(
        "--output-dir",
        "--output_dir",
        default=os.environ.get("SM_OUTPUT_DATA_DIR", str(Path("outputs"))),
    )
    p.add_argument(
        "--metrics-s3-uri",
        "--metrics_s3_uri",
        default="",
        help="s3://bucket/path.csv to append one row of run metrics per training job.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", "--img_size", type=int, default=224)
    p.add_argument("--batch-size", "--batch_size", type=int, default=64)
    p.add_argument("--num-workers", "--num_workers", type=int, default=2)

    p.add_argument("--max-epochs", "--max_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", "--min_delta", type=float, default=1e-4)

    p.add_argument("--lr-head", "--lr_head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", "--lr_backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", "--label_smoothing", type=float, default=0.05)

    p.add_argument(
        "--no-plots",
        "--no_plots",
        nargs="?",
        const=True,
        default=False,
        metavar="VAL",
        type=_no_plots_cli_value,
        help="Skip matplotlib figures (headless).",
    )
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
    args.label_smoothing = _hp("label_smoothing", float, args.label_smoothing)
    args.metrics_s3_uri = _hp("metrics_s3_uri", str, args.metrics_s3_uri)

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
        raise SystemExit("Provide --data-root")
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
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

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
    test_loss = float(nn.CrossEntropyLoss()(all_logits, all_y).item())
    topk = min(3, len(classes))
    topk_hits = all_logits.topk(topk, dim=1).indices.eq(all_y.view(-1, 1)).any(dim=1)
    test_topk_acc = float(topk_hits.float().mean().item())
    test_acc = float((preds == all_y).float().mean().item())
    print(f"test top-1 accuracy: {test_acc:.4f}")
    print(f"test top-{topk} accuracy: {test_topk_acc:.4f}")
    print(f"test loss: {test_loss:.4f}")

    print(f"METRIC test_top1_accuracy = {test_acc:.6f}")
    print(f"METRIC test_top3_accuracy = {test_topk_acc:.6f}")
    print(f"METRIC test_loss = {test_loss:.6f}")
    print(f"METRIC best_val_accuracy = {float(best_val):.6f}")
    print(f"METRIC best_val_loss = {float(min(history['val_loss'])):.6f}")
    print(f"METRIC final_train_loss = {float(history['train_loss'][-1]):.6f}")
    print(f"METRIC final_train_accuracy = {float(history['train_acc'][-1]):.6f}")
    print(f"METRIC final_val_loss = {float(history['val_loss'][-1]):.6f}")
    print(f"METRIC final_val_accuracy = {float(history['val_acc'][-1]):.6f}")
    print(f"METRIC train_seconds = {float(train_time):.3f}")
    print(f"METRIC epochs_ran = {float(epochs_ran)}")

    report = classification_report(
        all_y.numpy(), preds.numpy(), target_names=classes, digits=4, output_dict=True
    )
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    print(f"METRIC test_macro_f1 = {float(macro.get('f1-score', 0.0)):.6f}")
    print(f"METRIC test_weighted_f1 = {float(weighted.get('f1-score', 0.0)):.6f}")
    print(f"METRIC test_macro_precision = {float(macro.get('precision', 0.0)):.6f}")
    print(f"METRIC test_macro_recall = {float(macro.get('recall', 0.0)):.6f}")
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

    best_epoch_1idx = int(np.argmax(history["val_acc"]) + 1)
    sm_ctx = _extract_sm_context()
    meta = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "efficientnet_b0",
        "num_classes": len(classes),
        "img_size": args.img_size,
        "dataset_sizes": {"train": len(train_ds), "valid": len(valid_ds), "test": len(test_ds)},
        "max_epochs": args.max_epochs,
        "epochs_ran": epochs_ran,
        "best_epoch_1idx": best_epoch_1idx,
        "early_stopped": stopped_early,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "best_val_acc": float(best_val),
        "best_val_loss": float(min(history["val_loss"])),
        "batch_size": args.batch_size,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "train_seconds": train_time,
        "test_top1_acc": test_acc,
        "test_top3_acc": test_topk_acc,
        "test_loss": test_loss,
        "test_macro_f1": float(macro.get("f1-score", 0.0)),
        "test_weighted_f1": float(weighted.get("f1-score", 0.0)),
        "test_macro_precision": float(macro.get("precision", 0.0)),
        "test_macro_recall": float(macro.get("recall", 0.0)),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "runtime_context": sm_ctx,
        "metrics_s3_uri": args.metrics_s3_uri,
        "imagenet_mean": mean,
        "imagenet_std": std,
        "history": history,
        "final_metrics": {
            "final_train_loss": float(history["train_loss"][-1]),
            "final_train_accuracy": float(history["train_acc"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "final_val_accuracy": float(history["val_acc"][-1]),
        },
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

    run_metrics_row = {
        "created_utc": meta["created_utc"],
        "training_job_name": sm_ctx.get("training_job_name", ""),
        "model": meta["model"],
        "epochs_ran": epochs_ran,
        "best_epoch_1idx": best_epoch_1idx,
        "batch_size": args.batch_size,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "test_top1_acc": test_acc,
        "test_top3_acc": test_topk_acc,
        "test_loss": test_loss,
        "best_val_acc": float(best_val),
        "best_val_loss": float(min(history["val_loss"])),
        "test_macro_f1": float(macro.get("f1-score", 0.0)),
        "test_weighted_f1": float(weighted.get("f1-score", 0.0)),
        "train_seconds": float(train_time),
        "num_classes": len(classes),
        "img_size": args.img_size,
        "instance_type": sm_ctx.get("instance_type", ""),
        "device": meta["device"],
        "model_artifact_local": str(tar_path),
    }
    local_metrics_csv = out_dir / "run_metrics.csv"
    with open(local_metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_metrics_row.keys()))
        writer.writeheader()
        writer.writerow(run_metrics_row)
    print(f"wrote {local_metrics_csv}")

    if args.metrics_s3_uri:
        _append_metrics_csv_to_s3(args.metrics_s3_uri, run_metrics_row)

    print(f"wrote {tar_path} ({tar_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
