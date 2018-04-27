#!/bin/bash

docker build -t="mattmcd/mask_rcnn_lambda" docker
docker run --rm --mount type=bind,source=$PWD/build_artifacts,target=/app mattmcd/mask_rcnn_lambda:latest
rm -rf python/vendored/
unzip -q build_artifacts/tf_env.zip -d python/vendored
