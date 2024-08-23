#bin/bash

bentoml delete classification:v1.0
bentoml build --version v1.0
# for macos
bentoml containerize --opt platform=linux/amd64 classification:v1.0