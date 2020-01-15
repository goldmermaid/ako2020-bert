
ARG REGION
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/mxnet-inference:1.4.1-gpu-py3

RUN pip install --upgrade --user --pre 'mxnet-mkl' 'https://github.com/dmlc/gluon-nlp/tarball/v0.9.x'

RUN pip list | grep mxnet

COPY *.py /opt/ml/model/code/
COPY bert/data/qa.py /opt/ml/model/code/bert/data/
COPY bert/bert_qa_evaluate.py /opt/ml/model/code/bert/