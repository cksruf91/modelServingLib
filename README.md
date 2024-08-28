modelServingLib
----
| no | name                    | path         | status |
|---:|:------------------------|:-------------|:-------|
|  1 | Triton Inference Server | ./triton     | Done   |
|  2 | TorchServe              | ./torchserve | Done   |
|  3 | BentoML                 | ./bentoml    | Done   |

# Requirement
* python 3.11 이상
* Docker

# Summary
* 응답시간
  * 평균적인 응답시간은 __Triton__ 이 일관적으로 빠르다.
  * __Triton__(750ms) < __BentoML__(850ms) < __TorchServe__(1500ms, 실패)
    * 아래 BenchMark 참조
  * 1500ms 가 넘는 건에 대해서는 실패로 설정 했기 때문에 __TorchServe__ 는 fail 로 봐야 할듯
* 제약사항
  * __Triton__ 은 input 이나 output 이 라이브러리가 요구하는 특정 포멧을 맞춰야 하기 때문에 제약사항이 많음(내가 원하는 포멧의 아웃풋을 만들기 어려움) 반면 __TorchServe__, __BentoMl__ 은 데이터 형식에 대해서 좀더 자유로움
* 개인적으로 느낀 개발 난이도
  * __BentoML__ < __TorchServe__ < __Triton__
  * __BentoML__ 은 ServiceClass 만 정의하면 별다른 개발이 필요 없으며 config 도 같은 파일 내에서 할 수 있기 때문에 적용하기 쉬움
  * __TorchServe__ 도 HandlerClass 만 정의하면 되지만 mar 파일을 빌드하기 위해 고려해야 할 부분이 있어 진입하기 쉽지 않음
  * __Triton__, Model inference, PreProcess, PostProcess 모두 각각 별도 class 로 정의하고 이를 config.pbtxt 로 엮어야 하기 때문에 직관적 이지 않음


# BenchMark
## environment
* model: klue/roberta-base(huggingface) + pooling layer(output: 3)
* gpu: Tesla T4
* cuda 11.0
* torchserve==0.11.1
* torch==2.0.1
* bentoml==1.3.2
* triton
  * tritonclient==2.48.0
  * image: nvcr.io/nvidia/tritonserver:22.12-pyt-python-py3
* locust==2.31.2
  * users: 200 
  * spawn-rate: 5

## Result
| server                  | result                                                                                                           |
|:------------------------|:-----------------------------------------------------------------------------------------------------------------|
| Triton Inference Server | <img src="locust/benchmark/Triton-200-5-roberta.png" width="800px" height="500px" title="DynamicBatchTest"/>     |
| TorchServe              | <img src="locust/benchmark/torchserve-200-5-roberta.png" width="800px" height="500px" title="DynamicBatchTest"/> |
| BentoML                 | <img src="locust/benchmark/bentoml-200-5-roberta.png" width="800px" height="500px" title="DynamicBatchTest"/>    |
