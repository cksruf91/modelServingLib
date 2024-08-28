torch-model-archiver \
  --model-name cls_model \
  --version 1.0 \
  --serialized-file ml/model/model.pt \
  --handler cls_handler.py \
  --export-path model_store \
  -f \
  --requirements-file ../requirements.txt \
  --extra-files "ml/tokenizer/special_tokens_map.json,ml/tokenizer/tokenizer_config.json,ml/tokenizer/vocab.txt,ml/tokenizer/tokenizer.json"

torchserve --start --foreground \
  --model-store model_store \
  --models cls=cls_model.mar \
  --ts-config ./config/cpu_config.properties \
  --no-config-snapshots \
  --disable-token-auth