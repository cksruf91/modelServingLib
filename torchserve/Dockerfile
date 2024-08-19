FROM pytorch/torchserve:latest

RUN pip install --upgrade pip
RUN pip install transformers

ENTRYPOINT ["torchserve", "--start", "--foreground", "--model-store", "model_store", "--models", "cls=cls_model.mar", "--ts-config", "./config/cpu_config.properties", "--no-config-snapshots", "--disable-token-auth"]