from typing import List, Dict
import string
import unicodedata

import torch
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator
from ts.torch_handler.base_handler import BaseHandler


class MyHandler(BaseHandler):

    def __init__(self):
        super().__init__()
        self.tokenizer = get_tokenizer("basic_english")
        self.source_vocab = None
        self.device = torch.device('cpu')
        self.ngrams = 2

    def initialize(self, context):
        super().initialize(context)
        self.source_vocab = torch.load('/Users/changyeol/Project/personal_repo/torchserve/example/resource/source_vocab.pt')
        if self.source_vocab is None:
            raise ValueError("source vocab is None ")

    def preprocess(self, data: List[Dict]):
        line = data[0]
        text = line.get("data") or line.get("body")
        # Decode text if not a str but bytes or bytearray
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")
        text = text.lower()
        text = self._remove_accented_characters(text)
        text = self._remove_punctuation(text)
        tokens = self.tokenizer(text)
        text_tensor = torch.as_tensor(
            [self.source_vocab[token] for token in ngrams_iterator(tokens, ngrams=self.ngrams)],
            device=self.device,
        )
        return text_tensor, [text]

    def inference(self, data, *args, **kwargs):
        tensor, text = data
        offsets = torch.as_tensor([0], device=self.device)
        pred = self.model(tensor, offsets)
        return pred, text

    def postprocess(self, data):
        pred, text = data
        result = []
        for i in range(pred.size(0)):
            temp = {'text': text[i], 'pred': []}
            logit = F.softmax(pred)[i]
            ordered_index = torch.argsort(logit, descending=True).tolist()
            for idx in ordered_index:
                temp['pred'].append(
                    {self.mapping[str(idx)]: logit[idx].item()}
                )
            result.append(temp)
        return result

    @staticmethod
    def _remove_accented_characters(text: str):
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )

    @staticmethod
    def _remove_punctuation(text: str):
        return text.translate(str.maketrans("", "", string.punctuation))
