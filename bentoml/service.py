import logging
import string
from pathlib import Path
from typing import Dict, List

import bentoml
import torch
from transformers import AutoTokenizer

EXAMPLE_INPUT = (
    "The sun dips below the horizon, painting the sky orange.",
)
GPU_COUNT = torch.cuda.device_count()
if GPU_COUNT > 0:
    RESOURCE = {'gpu': GPU_COUNT, "memory": "1Gi"}
else:
    RESOURCE = {'cpu': 'cpu_count', "memory": "1Gi"}


@bentoml.service(
    resources=RESOURCE,
    workers=1,
    traffic={"timeout": 3},
)
class Classification:
    def __init__(self) -> None:
        self.logger = logging.getLogger('bentoml')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('ml/tokenizer')
        model_pt_path = Path('ml/model.pt')
        if not model_pt_path.exists():
            raise RuntimeError(f"Missing the model.pt file -> {model_pt_path.absolute()}")
        self.model = torch.jit.load(str(model_pt_path)).to(self.device)
        self.label = {i: s for i, s in enumerate(string.ascii_uppercase)}

    def _tokenize(self, text: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text=text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=100,
            return_token_type_ids=False
        ).to(self.device)

    @bentoml.api(
        batchable=True,
        max_batch_size=8,
        max_latency_ms=3000
    )
    def classification(self, text: List[str] = EXAMPLE_INPUT) -> List[Dict[str, Dict[str, float]]]:
        self.logger.info(f'text: {text}')
        tokens = self._tokenize(text)
        output = self.model(**tokens)
        result = []
        for array in output.detach().cpu().numpy():
            result.append(
                {'output': {self.label[idx]: score for idx, score in enumerate(array)}}
            )
        return result
