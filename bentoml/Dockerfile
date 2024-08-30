FROM python:3.11.9-slim
LABEL authors="prismo"

# sync local timezone
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# set up
RUN mkdir -p /home/app
WORKDIR /home/app

# install python libs
COPY requirements.txt .
RUN pip install -U pip
RUN pip install -r requirements.txt

ENTRYPOINT ["bentoml", "serve", "service:Classification"]