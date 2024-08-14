torch-model-archiver \
  --model-name embedding_model \
  --version 1.0 \
  --serialized-file ml/model/model.pt \
  --handler embedding_handler.py \
  --export-path model_store \
  -f \
  --requirements-file ../requirements.txt \
  --extra-files "ml/tokenizer/special_tokens_map.json,ml/tokenizer/tokenizer_config.json,ml/tokenizer/vocab.txt"

torchserve --start --foreground \
  --model-store model_store \
  --models embedding=embedding_model.mar \
  --no-config-snapshots \
  --disable-token-auth