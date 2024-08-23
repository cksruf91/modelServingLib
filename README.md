modelServingLib
----


| no | name                    | path         | status       |
|---:|:------------------------|:-------------|:-------------|
|  1 | Triton inference Server | ./triton     | Done         |
|  2 | TorchServe              | ./torchserve | Done         |
|  3 | bentoml                 | ./bentoml    | in-progress  |

# Requirement
## python
* python 3.9 이상
## docker
version 정보
```text
Client:
 Version:           27.0.3
 API version:       1.46
 Go version:        go1.21.11
 Git commit:        7d4bcd8
 Built:             Fri Jun 28 23:59:41 2024
 OS/Arch:           darwin/arm64
 Context:           desktop-linux

Server: Docker Desktop 4.32.0 (157355)
 Engine:
  Version:          27.0.3
  API version:      1.46 (minimum version 1.24)
  Go version:       go1.21.11
  Git commit:       662f78c
  Built:            Sat Jun 29 00:02:44 2024
  OS/Arch:          linux/arm64
  Experimental:     false
 containerd:
  Version:          1.7.18
  GitCommit:        ae71819c4f5e67bb4d5ae76a6b735f29cc25774e
 runc:
  Version:          1.7.18
  GitCommit:        v1.1.13-0-g58aa920
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0

```