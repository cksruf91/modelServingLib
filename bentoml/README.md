BentoML
---

# Step 1: Export the model
model 및 tokenizer 생성
```shell
# 프로젝트 root dir 에서
python export.py --bentoml
```

# Step 2: Set Up Files
`service.py` 파일 생성(파일 이름은 무관함)

```python
from typing import List, Dict

import bentoml


@bentoml.service(
  resources={"cpu": "2", "memory": "500MiB", "gpu": 1},
  workers=2,
  traffic={"timeout": 10},
)
class Classification:

  def __init__(self) -> None:
    self.tokenizer = ...
    self.model = ...

  @bentoml.api(
    route="/custom/url/name",
    batchable=True,
    batch_dim=(0, 0),
    max_batch_size=32,
    max_latency_ms=1000
  )
  def classification(self, text: List[str], ctx: bentoml.Context) -> List[Dict]:
    _ = ctx.request.headers  # request header 접근
    tokens = self.tokenizer(text)
    output = self.model(**tokens)
    ctx.response.status_code = 200  # status_code, header, cookies 등 접근
    return [{"output": output}]
```
* __bentoml.service__ : resource(cpu, gpu, mem), worker, timeout(request timeout) 등을 설정 할 수 있다(cpu와 gpu는 같이 세팅할 수 없다. 위는 예시)
* __bentoml.api__
  * __route__ : api endpoint 이름은 기본적으로 method name 으로 정해지나 route 설정으로 변경 할 수 있다.
  * __batchable__ : Adaptive Batching(Dynamic Batching) 설정을 끄고 켤 수 있음
    * Adaptive Batching 을 활성화 시켰다면 batch 처리가 가능하도록 input, output 모두 List, np.ndarray 같은 concat 가능한 형태여야 한다.
  * __max_batch_size__ : 최대 배치 사이즈
  * __max_latency_ms__ : 최대 지연 길이, 배치가 처리되는 최대 시간을 의미, 배치가 `max_latency_ms` 안에 처리되지 못할때 503 에러를 발생시킨다. 실제 batch input 을 기다리는 시간은 요청량에 따라 유동적으로 조절된다. 
  * __batch_dim__ : batch 로 들어오는 데이터가 concat 되는 dimension 을 의미 (input dim, output dim)

# Step 3: Run API
bentoml 이 설치 되었고 serviceClass 가 적절하게 설정 되어 있다면 아래 명령어로 service 를 생성할 수 있음
```shell
bentoml serve service:Classification
```
* `bentoml serve {FileName}:{ClassName}` 형태

# Step 4: Build and Running with Docker

## BentoML docker image 사용 방법
### Build Bento 
bentoml build 명령어는 디렉토리의 `bentofile.yaml` 을 참조한다.
```shell
# 이미 같은 버전으로 빌드된 파일이 있을 경우 제거
bentoml delete classification:v1.0

bentoml build --version v1.0 
```
* __--version__ : model 의 버전이자 docker tag로 사용된다.
* Build 된 파일은 기본적으로 `~/bentoml/bentos`에 위치 하게 된다.
* 현재 빌드 되어 있는 bento 파일이 어떤게 있는지 궁금할 경우 `bentoml list` 명령어를 통해 확인 할 수 있다.

### Containerize and Running 
```shell
# Linux
bentoml containerize classification:v1.0
# for MacOS
bentoml containerize --opt platform=linux/amd64 classification:v1.0 

# run container
docker run --rm -p 3000:3000 classification:v1.0
```

## [Optional] Custom Docker Image
* 기본 bentoml docker 이미지가 cuda 11.0 버전을 지원하지 않아 직접 이미지 생성
```shell
cp ../requirements.txt .
docker build . -t classification:v1.0
docker run --rm -p 3000:3000 -v ${PWD}:/home/app classification:v1.0
# For gpu
docker run --gpus all --rm -p 3000:3000 -v ${PWD}:/home/app classification:v1.0
```

# Step 5: Test API
* swagger ui : http://localhost:3000/

via curl
```shell
curl -X 'POST' \
  'http://localhost:3000/classification' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": [
    "The sun dips below the horizon, painting the sky orange."
  ]
}'
```


