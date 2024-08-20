TorchServe
---

# Step 1: Export the model
```shell
python export.py --torchserve
```
# Step 2: Set Up
## Define Handler & Build Mar
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
        data: List[Dict] -> ex) [{data: {request body}}] 
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
  --model-name cls_model \
  --version 1.0 \
  --serialized-file ml/model/model.pt \
  --handler embedding_handler.py \
  --export-path model_store \
  -f \
  --requirements-file ../requirements.txt \
  --extra-files "ml/tokenizer/special_tokens_map.json,ml/tokenizer/tokenizer_config.json,ml/tokenizer/vocab.txt"
```
* [--model-name] : *.mar 파일의 이름이 됨
* [--export-path] : mar 파일 저장 위치
* [-f] : 이미 mar 파일이 존재할 경우 강제로 업데이트 
* [extra-file] : handler 구동을 위해 필요한 파일들을 지정 해당 파일들은 context.system_properties.get("model_dir") 경로에 위치하게 된다.

## Dynamic Batching 설정
`config.properties` 에 각 모델 별로 `batchSize` 와 `maxBatchDelay` 를 설정 하여 Dynamic batching 을 사용 할 수 있다.
```shell
# config.properties 파일
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
....
models={\
  "embedding": {\
    "1.0": {\
        "defaultVersion": true,\
        "marName": "Embedding.mar",\
        "minWorkers": 7,\
        "maxWorkers": 7,\
        "batchSize": 8,\
        "maxBatchDelay": 300,\
        "responseTimeout": 120\
    }\
  }\
}
```

# Step 3: Running Server
torchserve api 실행
```shell
# cpu 사용시
torchserve --start --foreground \
  --model-store model_store \
  --models cls=cls_model.mar \
  --ts-config ./config/cpu_config.properties \
  --no-config-snapshots \
  --disable-token-auth

# gpu 사용시
torchserve --start --foreground \
  --model-store model_store \
  --models cls=cls_model.mar \
  --ts-config ./config/gpu_config.properties \
  --no-config-snapshots \
  --disable-token-auth
```

# Step 4: Build and Running with Docker
mar 파일 빌드 후 
* cpu
```shell
docker build -t my_torch_serve:1.0 .

docker run --rm -p 8080:8080 -p 8081:8081 \
  -v ${PWD}:/home/model-server \
  my_torch_serve:1.0
```
* cuda(11.x)
```shell
docker build -f Dockerfile.gpu . -t my_torch_serve:1.0

docker run --gpus all --rm -p 8080:8080 -p 8081:8081 \
  --shm-size=1G \
  -v ${PWD}:/home/model-server \
  my_torch_serve:1.0
```
* Cuda docker image 가 메모리를 많이 사용하기 때문에 종종 shared memory 부족으로 서비스를 띄우지 못하는 일이 발생한다 `--shm-size` 옵션을 지정하면 해결 할 수 있다.

# Step 5: API test
## curl 사용
model inference 방법
```shell
curl -X POST http://127.0.0.1:8080/predictions/cls -H "Content-Type: application/json" --data  "{\"text\": \"sample\"}"
```
아래 endpoint 를 통해 model 정보를 확인 할 수 있다.
```shell
curl http://localhost:8081/models/cls
```
