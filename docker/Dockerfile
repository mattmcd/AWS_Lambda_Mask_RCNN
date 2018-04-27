FROM amazonlinux:latest

MAINTAINER Matt McDonnell "matt@matt-mcdonnell.com"

RUN yum -y -q update

RUN yum -y -q install python3.5 python3-pip zip python35-setuptools && \
  easy_install-3.5 pip && \
  pip3 install --no-cache-dir --upgrade pip setuptools

RUN pip3 install -q virtualenv

RUN virtualenv -p python3 tf_env

RUN source tf_env/bin/activate && \
  pip3 install -q --upgrade numpy tensorflow keras

RUN cd tf_env/lib/python3.5/site-packages/ && zip -qr ~/tf_env.zip . --exclude \*.DS_Store \*.pyc /external/\* /tensorflow/contrib/\* /tensorflow/include/unsupported/\*

CMD cp ~/tf_env.zip /app/