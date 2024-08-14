FROM pytorch/torchserve:latest

COPY . .

ENTRYPOINT ["torchserve", "--start", "--foreground", "--model-store", "model_store", "--models", "em=Embedding.mar", "--no-config-snapshots", "--disable-token-auth"]