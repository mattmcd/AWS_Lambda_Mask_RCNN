#!/bin/bash

rm build_artifacts/mask_rcnn_lambda.zip
cd python; zip -qr ../build_artifacts/mask_rcnn_lambda *; cd ..