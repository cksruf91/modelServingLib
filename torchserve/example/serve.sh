torch-model-archiver --model-name my_text_classifier \
  --version 1.1 \
  --model-file model/model.py \
  --serialized-file resource/model.pt \
  --handler my_handler \
  --extra-files "resource/index_to_name.json,resource/source_vocab.pt"

mkdir model_store
mv my_text_classifier.mar model_store/

torchserve --start --foreground \
  --model-store model_store \
  --models tc=my_text_classifier.mar \
   --no-config-snapshots

# curl -X POST http://127.0.0.1:8080/predictions/tc -T example/sample_text.txt
