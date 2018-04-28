FROM amazonlinux:latest

MAINTAINER Matt McDonnell "matt@matt-mcdonnell.com"

RUN yum -y -q update

RUN yum -y -q install python3.6 python3-pip zip python36-setuptools && \
  easy_install-3.6 pip && \
  pip3 install --no-cache-dir --upgrade pip setuptools

RUN pip3 install -q virtualenv

RUN virtualenv -p python3 tf_env

RUN source tf_env/bin/activate && \
  pip3 install -q --upgrade numpy scikit-image tensorflow keras

RUN cd tf_env/lib/python3.6/site-packages/ && zip -qr ~/tf_env.zip . \
  --exclude \*.DS_Store \*.pyc /external/\* /tensorflow/contrib/\* /tensorflow/include/unsupported/\* \
  /tensorboard/\* /tensorboard-1.8.0.dist-info/\*  /networkx/\* /networkx-2.1.dist-info/\*

CMD cp ~/tf_env.zip /app/
