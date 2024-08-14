import argparse
import warnings
from pathlib import Path

import torch
from transformers import DistilBertTokenizer, DistilBertModel


class Args(argparse.ArgumentParser):

    def __init__(self):
        super(Args, self).__init__()
        self.add_argument('-a', '--all', action='store_true')
        self.add_argument('-tn', '--triton', action='store_true')
        self.add_argument('-ts', '--torchserve', action='store_true')
        self.add_argument('-bt', '--bentoml', action='store_true')
        self._ns: argparse.Namespace = self.parse_args()
        self.all: bool = self._ns.all
        self.triton: bool = self._ns.triton
        self.torchserve: bool = self._ns.torchserve
        self.bentoml: bool = self._ns.bentoml


class ModelBuilder:

    def __init__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'------------------- device : {str(device)} ------------------------')
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'monologg/distilkobert',
            clean_up_tokenization_spaces=False
        )

        self.model = (
            DistilBertModel
            .from_pretrained("monologg/distilkobert", torchscript=True)
            .to(device)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            example_text = "Replace me by any text you'd like."
            encoded_input = self.tokenizer(example_text, return_tensors='pt').to(device)
            self.traced_model = torch.jit.trace(
                self.model, [encoded_input['input_ids'], encoded_input['attention_mask']]
            )

    def save_model(self, path: str):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.jit.save(self.traced_model, str(path))
        print(f"save model -> {path.absolute()}")

    def save_tokenizer(self, path: str):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        self.tokenizer.save_pretrained(str(path))
        print(f"save tokenizer -> {path.absolute()}")


if __name__ == '__main__':
    arg = Args()
    builder = ModelBuilder()
    if arg.triton or arg.all:
        builder.save_model("./triton/model_repository/distilbert/1/model.pt")
        builder.save_tokenizer("./triton/model_repository/tokenizer/resource")
    if arg.torchserve or arg.all:
        builder.save_model("./torchserve/ml/model/model.pt")
        builder.save_tokenizer("./torchserve/ml/tokenizer")
