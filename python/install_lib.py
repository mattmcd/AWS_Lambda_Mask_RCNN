import os
import zipfile
import boto3
import sys
from subprocess import call

def library_install():
    vendored_file = 'numpy_skimage_tf_keras_lambda.zip'
    model_file = 'mask_rcnn_coco.h5'
    DATA_DIR = '/tmp'
    LAMBDA_LIB_BUCKET = 'mattmcd-lambda-libraries'
    S3_REGION_NAME = 'eu-west-1'
    local_vendored = os.path.join(DATA_DIR, vendored_file)
    local_model = os.path.join(DATA_DIR, model_file)

    LIB_DIR = os.path.join(DATA_DIR, 'vendored')

    if not os.path.isfile(local_model):
        files = os.listdir("/tmp")
        if len(files) > 0:
            print('Cleaning /tmp directory: removing {} files'.format(len(files)))
            call('rm -rf /tmp/*', shell=True)

        print('Connecting to S3')
        s3 = boto3.resource('s3', region_name=S3_REGION_NAME)
        bucket = s3.Bucket(LAMBDA_LIB_BUCKET)
        print('Copying vendored libraries to local /tmp')
        bucket.download_file(vendored_file, local_vendored)
        print('Libraries copied to local /tmp')
        zipref = zipfile.ZipFile(local_vendored)
        zipref.extractall(DATA_DIR)
        zipref.close()
        print('Appending libraries to path')
        sys.path.append(DATA_DIR)
        print(sys.path)
        print('Appended libraries to path')

        print('Delete zipfile')
        try:
            os.remove(local_vendored)
        except OSError:
            pass

        print('Copying model to local /tmp')
        bucket.download_file(model_file, local_model)
        print('Model downloaded')

        print('Copying demo data')
        bucket.download_file('cows_800x600.jpg', os.path.join(DATA_DIR, 'cows_800x600.jpg'))
    else:
        print('Libaries already copied locally')

    if LIB_DIR not in sys.path:
        sys.path.append(LIB_DIR)

    return DATA_DIR, local_model
