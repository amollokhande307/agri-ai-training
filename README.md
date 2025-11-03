\# Agri AI Training - minimal starter



A minimal PyTorch training repository set up for CI/CD with AWS CodeBuild and SageMaker.



\## Steps



1\.  \*\*Create S3 bucket\*\* and upload datasets (if you have real data).

2\.  \*\*Ensure AWS is configured\*\* (CLI or CodeBuild environment variables).

3\.  Push this repository to \*\*GitHub\*\*.

4\.  In CodeBuild (or local), run `python trigger\_training.py --bucket <bucket> --role <role\_arn> --region <region> --launch`.

5\.  Fill \*\*TrainingImage\*\* in the printed AWS CLI command or use the SageMaker SDK to run an estimator.



---



\## For testing



To ensure the training code works before deploying:



1\.  \*\*Install dependencies\*\* (if you haven't already):

&nbsp;   ```bash

&nbsp;   pip install -r train\\requirements.txt

&nbsp;   ```



2\.  \*\*Run locally\*\* with synthetic data:

&nbsp;   ```bash

&nbsp;   python train\\train.py --use\_synthetic --epochs 2

&nbsp;   ```

