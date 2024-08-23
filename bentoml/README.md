BentoML
---

# Step 1: Export the model
```shell
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
* __bentoml.service__ : resource(cpu, gpu, mem), worker, timeout(request timeout) 등을 설정 할 수 있다
* __bentoml.api__
  * __route__ : api endpoint 이름은 기본적으로 method name 으로 정해지나 route 설정으로 변경 할 수 있다.
  * __batchable__ : Adaptive Batching(Dynamic Batching) 설정을 끄고 켤 수 있음
    * Adaptive Batching 을 활성화 시켰다면 batch 처리가 가능하도록 input, output 모두 List, np.ndarray 같은 concat 가능한 형태여야 한다.
  * __batch_dim__ : batch 로 들어오는 데이터가 concat 되는 dimension 을 의미 (input dim, output dim)

# Step 3: Run API
```shell
bentoml serve service:Classification
```

# Step 4: Build and Running with Docker
bentoml build 명령어는 디렉토리의 `bentofile.yaml` 을 참조한다.
```shell
bentoml delete classification:v1.0
bentoml build --version v1.0 
# for macos
bentoml containerize --opt platform=linux/amd64 classification:v1.0 

# run container
docker run --rm -p 3000:3000 classification:v1.0
```
* __--version__ : model 의 버전이자 docker tag로 사용된다.
* 

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


