import os
import string
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler


class EmbeddingHandler(BaseHandler):

    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False
        self.model = None
        self.tokenizer = None
        self.label = {i: s for i, s in enumerate(string.ascii_uppercase)}

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        Args:
            context: context contains model server system properties
        """
        self.manifest = context.manifest
        self.context = context
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.initialized = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess(self, data: List[Dict]) -> Dict[str, torch.Tensor]:
        line = data[0]
        data = line.get("data") or line.get("body")
        text = data.get('text')
        print(f'request: {text}')
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")

        tokens: Dict[str, torch.Tensor] = self.tokenizer(
            text=text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=100,
            return_token_type_ids=False
        ).to(self.device)
        return tokens

    def inference(self, data, *args, **kwargs):
        pred = self.model(**data)
        return pred

    def postprocess(self, pred) -> List[Dict]:
        pred = pred.detach().cpu().numpy().astype(np.float64)  # float32 는 json 변환이 안됨
        print(f"output : {pred.shape}")
        indices = np.argsort(pred, axis=1)
        classes = list(np.vectorize(self.label.get)(indices))  # index -> label mapping
        values = list(np.sort(pred, axis=1))

        response = []
        for i in range(pred.shape[0]):
            response.append(
                {c: v for c, v in zip(classes[i], values[i])}
            )
        return response

    def handle(self, data, context):
        data = self.preprocess(data)
        pred = self.inference(data, context)
        return self.postprocess(pred)

    def _load_model(self):
        """ Read model serialize/pt file """
        model_dir = self.context.system_properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']

        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError(f"Missing the model.pt file -> {model_pt_path}")
        return torch.jit.load(model_pt_path).to(self.device)

    def _load_tokenizer(self):
        model_dir = self.context.system_properties.get("model_dir")
        return AutoTokenizer.from_pretrained(model_dir)
