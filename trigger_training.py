# trigger_training.py
import os
import tarfile
import boto3
import time
import argparse
from pathlib import Path

# You can use the sagemaker SDK instead (example commented)
# import sagemaker
# from sagemaker.pytorch import PyTorch

def make_source_tar(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    print("Created tar:", output_filename)

def upload_to_s3(bucket, key, filename):
    s3 = boto3.client("s3")
    s3.upload_file(filename, bucket, key)
    s3_uri = f"s3://{bucket}/{key}"
    print("Uploaded:", s3_uri)
    return s3_uri

def main(args):
    bucket = args.bucket
    region = args.region
    role = args.role  # must be SageMaker execution role ARN
    job_name = f"agri-train-{int(time.time())}"
    source_dir = "train"
    tar_name = "train-source.tar.gz"

    # package
    make_source_tar(source_dir, tar_name)
    key = f"sage-source/{tar_name}"
    s3_uri = upload_to_s3(bucket, key, tar_name)

    if args.launch:
        # Minimal create_training_job via boto3 (example).
        # NOTE: using prebuilt container images is region/version dependent.
        # We use the SageMaker SDK flow recommended in production; here is a simple boto3 example:
        sagemaker = boto3.client("sagemaker", region_name=region)

        # You MUST replace TrainingImage with a valid PyTorch image for your region,
        # or use the SageMaker Python SDK Estimator which picks image automatically.
        # For safety we will present the user with the CLI command to run.
        print("\n--- SageMaker training job INFO ---")
        print("We did not launch the job automatically because container image URIs vary by region.")
        print("Run the AWS CLI command below (fill in TrainingImage for your region) OR use the SageMaker SDK.")
        print()
        print("aws sagemaker create-training-job \\")
        print(f"  --region {region} \\")
        print(f"  --training-job-name {job_name} \\")
        print('  --algorithm-specification TrainingImage=<PUT_PYTORCH_IMAGE_URI_HERE>,TrainingInputMode=File \\')
        print(f"  --role-arn {role} \\")
        print("  --input-data-config '[{\"ChannelName\":\"training\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"%s\",\"S3DataDistributionType\":\"FullyReplicated\"}}}]' \\" % s3_uri)
        print("  --output-data-config 'S3OutputPath=s3://%s/sagemaker-output/' \\" % bucket)
        print("  --resource-config '{\"InstanceType\":\"ml.m5.large\",\"InstanceCount\":1,\"VolumeSizeInGB\":30}' \\")
        print("  --stopping-condition 'MaxRuntimeInSeconds=86400'")
        print()
        print("Or install the SageMaker Python SDK and use a PyTorch Estimator to launch the job programmatically.")
    else:
        print("Uploaded source only. Use --launch to print instructions to start a SageMaker job.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="S3 bucket to upload source")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--launch", action="store_true", help="Print instructions to launch training job")
    args = parser.parse_args()
    main(args)
