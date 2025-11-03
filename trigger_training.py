import boto3, time, os, json

region = os.environ.get("AWS_REGION", "ap-south-1")
sagemaker = boto3.client('sagemaker', region_name=region)

job_name = "agri-ai-train-" + str(int(time.time()))

# Example training image - CPU PyTorch managed image (no docker build required).
training_image = "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.0.0-cpu-py310"

role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
input_s3 = os.environ.get("S3_TRAINING_URI")
output_s3 = os.environ.get("S3_OUTPUT_URI")

print("Starting job:", job_name)
response = sagemaker.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={
        'TrainingImage': training_image,
        'TrainingInputMode': 'File'
    },
    RoleArn=role_arn,
    InputDataConfig=[
        {
            'ChannelName': 'training',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': input_s3,
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        }
    ],
    OutputDataConfig={'S3OutputPath': output_s3},
    ResourceConfig={
        'InstanceType': 'ml.m5.large',
        'InstanceCount': 1,
        'VolumeSizeInGB': 20
    },
    StoppingCondition={'MaxRuntimeInSeconds': 3600},
    HyperParameters={
        'epochs': '5',
        'batch_size': '32'
    }
)

print("SageMaker response:", json.dumps(response, indent=2))
