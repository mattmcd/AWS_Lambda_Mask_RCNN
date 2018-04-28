#!/bin/bash

docker build -t="mattmcd/mask_rcnn_lambda" docker

# Create tf_env.zip in build_artifacts - this is library file needed by Lambda
docker run --rm --mount type=bind,source=$PWD/build_artifacts,target=/app mattmcd/mask_rcnn_lambda:latest

# Copy library to S3
aws s3 cp build_artifacts/tf_env.zip s3://mattmcd-lambda-libraries/numpy_skimage_tf_keras_lambda.zip

# Create local unzipped copy of library for testing in PyCharm
rm -rf build_artifacts/vendored/
unzip -q build_artifacts/tf_env.zip -d build_artifacts/vendored
