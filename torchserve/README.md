TorchServe
---

# Step 1: Export the model
```shell
python export.py --torchserve
```
# Step 2: Set Up
## Handler and Build Mar
custom handler 생성 [참고](https://pytorch.org/serve/custom_service.html)   
Handler class 는 모듈 가장 상단에 있어야 한다.
```python
from ts.torch_handler.base_handler import BaseHandler

class MyHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None
        self.tokenizer = None
    
    def initialize(self, context):
        """ model, tokenizer loading 및 초기화시 필요한 동작 """
    def preprocess(self, data):
        """ 데이터 전처리 정의
        data: List[Dict] -> ex) [{data: {request data}}] 
        """
    def inference(self, data, *args, **kwargs):
        """ model inference 정의 """
    def postprocess(self, pred):
        """ 후처리 정의 """
    def handle(self, data, context):
        """ handling model, inference 시 호출되는 부분
        """
```
mar 파일 빌드
```shell
torch-model-archiver \
  --model-name embedding_model \            <- *.mar 파일의 이름이 됨
  --version 1.0 \
  --serialized-file ml/model/model.pt \ 
  --handler embedding_handler.py \
  --export-path model_store \               <- mar 파일 저장 위치
  -f \                                      <- 이미 mar 파일이 존재할 경우 강제로 업데이트
  --requirements-file ../requirements.txt \
  --extra-files "ml/tokenizer/special_tokens_map.json,ml/tokenizer/tokenizer_config.json,ml/tokenizer/vocab.txt"
                                            <- extra-file 은 실행시 context.system_properties.get("model_dir") 에서 위치하게 된다
```

# Step 3: Running Server
torchserve api 실행
```shell
torchserve --start --foreground \
  --model-store model_store \
  --models embedding=embedding_model.mar \
  --no-config-snapshots \
  --disable-token-auth
```

# Step 4: Build and Running with Docker
```shell
cp ../requirements.txt .
docker build -t my_torch_serve:1.0 .

docker run --rm -p 8080:8080 -p 8081:8081 my_torch_serve:1.0
docker rmi $(docker images | grep none | awk '{print $3}')
```

# Step 5: Test API
## curl 사용
model inference 방법
```shell
curl -X POST http://127.0.0.1:8080/predictions/embedding -H "Content-Type: application/json" --data  "{\"text\": \"sample\"}"
```
아래 endpoint 를 통해 model 정보를 확인 할 수 있다.
```shell
curl http://localhost:8081/models/embedding
```
