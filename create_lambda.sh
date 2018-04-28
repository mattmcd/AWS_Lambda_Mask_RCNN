#!/bin/bash

rm build_artifacts/mask_rcnn_lambda.zip
cd python; zip -qr ../build_artifacts/mask_rcnn_lambda *; cd ..
aws s3 cp build_artifacts/mask_rcnn_lambda.zip s3://mattmcd-lambda-libraries/mask_rcnn_lambda.zip
aws lambda create-function --function-name mask_rcnn_lambda --role arn:aws:iam::992910104567:role/lambda_basic_execution --runtime python3.6 --handler handler.predict --code S3Bucket=mattmcd-lambda-libraries,S3Key=mask_rcnn_lambda.zip
