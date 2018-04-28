#!/bin/bash

rm build_artifacts/mask_rcnn_lambda.zip
cd python; zip -qr ../build_artifacts/mask_rcnn_lambda *; cd ..
aws s3 cp build_artifacts/mask_rcnn_lambda.zip s3://mattmcd-lambda-libraries/mask_rcnn_lambda.zip
aws lambda update-function-code --function-name mask_rcnn_lambda --s3-bucket mattmcd-lambda-libraries --s3-key mask_rcnn_lambda.zip