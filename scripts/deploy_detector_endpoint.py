from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-data", required=True, help="S3 URI to detector model.tar.gz (best.pt + metadata)")
    p.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    p.add_argument("--endpoint-name", required=True, help="Endpoint name")
    p.add_argument("--instance-type", default="ml.g4dn.xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-1"))
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    boto_sess = boto3.Session(region_name=args.region)
    sm_sess = sagemaker.Session(boto_session=boto_sess)

    model = PyTorchModel(
        model_data=args.model_data,
        role=args.role_arn,
        entry_point="inference_detect.py",
        source_dir=str(repo_root / "src"),
        framework_version="2.2",
        py_version="py310",
        sagemaker_session=sm_sess,
    )

    predictor = model.deploy(
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        initial_instance_count=args.instance_count,
        wait=True,
    )
    print(f"Detector endpoint ready: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
