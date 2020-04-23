FROM python:3.8.2

WORKDIR /workspace/
COPY . /workspace/PySwallow

RUN apt-get update && apt-get install -y \
  vim \
  git

RUN pip install -r /workspace/PySwallow/requirements.txt
