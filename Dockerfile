FROM continuumio/anaconda3:2020.07

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN conda install -y tensorflow \
    && pip install keras-rectified-adam slack_sdk tqdm

WORKDIR /app