Triton inference server Quick start guide
-----
Form [Nvidia Triton quick start guide](https://github.com/triton-inference-server/tutorials/blob/main/Quick_Deploy/PyTorch/README.md)

# Step 1: Export the model
모델을 만들기 위해 docker 로 환경을 구축
```shell
# for cpu
docker run -it --rm -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:24.01-py3
# for gpu
docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:24.01-py3
# Resnet50 model create
python export.py
```
실행하면 ```model.pt``` 파일을 얻을 수 있음
> **Note**
> ```nvcr.io/nvidia/pytorch:24.01-py3``` 이미지 겁나 무거움 19G

# Step 2: Set Up Triton Inference Server
Triton 서버를 만들어 보자
```text
model_repository
|
+-- resnet50
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.pt
```
위와같은 구조의 파일이 필요
```shell
# for cpu
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# for gpu
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:<xx.yy>-py3 \
  tritonserver \
  --model-repository=/models
```

# Step 3: Using a Triton Client to Query the Server
inference 에 필요한 이미지 다운로드
```shell
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg" 
```

local 에서 inference 실행
```shell
python client_inference.py
```

Docker 환경에서 Inference 실행 하고 싶다면
```bash
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:24.01-py3-sdk bash
pip install torchvision opencv-python
python client_inference.py
```
> **Note**
> ```nvcr.io/nvidia/tritonserver:24.01-py3-sdk``` 이미지도 상당히 무겁기 때문에 추천 하지 않음
