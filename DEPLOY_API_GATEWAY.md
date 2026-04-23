# Deploy Card Classifier Behind API Gateway

This document captures the working flow for:

- SageMaker endpoint inference
- Lambda adapter
- API Gateway route
- Endpoint swapping without changing API Gateway

Region used: `eu-west-1`  

## Deploying a SageMaker endpoint

From repo root:

```bash
cd /home/sagemaker-user/BlackjackProject

python scripts/deploy_endpoint.py \
  --model-data "s3://22355359-ml-noa-cs5042/training/<job>/<job>/output/model.tar.gz" \
  --role-arn "arn:aws:iam::403903769410:role/service-role/AmazonSageMakerAdminIAMExecutionRole" \
  --endpoint-name "cards-classifier-dev-02" \
  --instance-type "ml.m5.large" \
  --instance-count 1 \
  --region "eu-west-1"
```
# (here, it was called "cards-classifier-dev-02", but I did update the -dev-xx for each time I started a new one, just to keep track for myself)
Check status:

```bash
aws sagemaker describe-endpoint \
  --region eu-west-1 \
  --endpoint-name cards-classifier-dev-02 \
  --query EndpointStatus \
  --output text
```

Expected: `InService`

## Configuring active endpoint in SSM

Seting active endpoint: 

```bash
aws ssm put-parameter \
  --region eu-west-1 \
  --name /blackjack/active-endpoint \
  --type String \
  --overwrite \
  --value cards-classifier-dev-02
```

Verify:

```bash
aws ssm get-parameter \
  --region eu-west-1 \
  --name /blackjack/active-endpoint \
  --query Parameter.Value \
  --output text
```

## Lambda setup

Code file: `api/lambda_predict.py`

If using inline editor, paste file contents into `lambda_function.py`.  
Handler should be:

- `lambda_function.lambda_handler` (for inline `lambda_function.py`), or
- `lambda_predict.lambda_handler` (if uploaded as separate file in zip)

Environment variables:

- `ACTIVE_ENDPOINT_PARAM=/blackjack/active-endpoint`
- Optional fallback: `SAGEMAKER_ENDPOINT=<endpoint-name>`  
  (If set, it overrides SSM lookup.)

##  Lambda IAM permissions

Lambda execution role needs:

- `sagemaker:InvokeEndpoint`
- `ssm:GetParameter`

Example policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowInvokeDevEndpoints",
      "Effect": "Allow",
      "Action": ["sagemaker:InvokeEndpoint"],
      "Resource": [
        "arn:aws:sagemaker:eu-west-1:403903769410:endpoint/cards-classifier-dev-*"
      ]
    },
    {
      "Sid": "AllowReadActiveEndpointParam",
      "Effect": "Allow",
      "Action": ["ssm:GetParameter"],
      "Resource": [
        "arn:aws:ssm:eu-west-1:403903769410:parameter/blackjack/active-endpoint"
      ]
    }
  ]
}
```

## API Gateway setup

Use HTTP API:

1. Create API -> HTTP API
2. Integration: Lambda (`cards-predict-lambda`)
3. Route: `POST /predict`
4. Stage: `$default` (auto-deploy enabled)
5. Enable CORS if needed (`POST, OPTIONS`, `content-type`)

Invoke URL pattern:

```text
https://<api-id>.execute-api.eu-west-1.amazonaws.com/predict
```

## Test request

Request body:

```json
{
  "image_base64": "<BASE64_IMAGE>",
  "top_k": 3
}
```

Expected response shape:

```json
{
  "endpoint": "cards-classifier-dev-02",
  "predictions": [
    {
      "top_prediction": {
        "label": "AH",
        "probability": 0.98,
        "class_index": 0
      },
      "top_k": [
        {"label": "AH", "probability": 0.98, "class_index": 0}
      ]
    }
  ]
}
```

## Swaping a model later (no API Gateway changes)

1. Deploy a new endpoint (example: `cards-classifier-dev-03`)
2. Update SSM parameter:

```bash
aws ssm put-parameter \
  --region eu-west-1 \
  --name /blackjack/active-endpoint \
  --type String \
  --overwrite \
  --value cards-classifier-dev-03
```

All new API requests route to the new endpoint automatically.

## Cost control

Endpoints bill while `InService`. I deleted them when not needed:

```bash
aws sagemaker delete-endpoint \
  --region eu-west-1 \
  --endpoint-name cards-classifier-dev-02
```

