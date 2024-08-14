import os
from typing import List, Dict

import torch
from transformers import DistilBertTokenizer
from ts.torch_handler.base_handler import BaseHandler


class EmbeddingHandler(BaseHandler):

    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False
        self.model = None
        self.tokenizer = None

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
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")

        tokens: Dict[str, torch.Tensor] = self.tokenizer(
            text=text, return_tensors='pt', padding=True
        ).to(self.device)
        return tokens

    def inference(self, data, *args, **kwargs):
        pred = self.model(**data)
        return pred

    def postprocess(self, pred):
        return pred[0].detach().cpu().numpy().tolist()

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
        return DistilBertTokenizer.from_pretrained(model_dir)
