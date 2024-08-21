import argparse
import warnings
from pathlib import Path

import torch
from transformers import ElectraModel, ElectraTokenizer


class Args(argparse.ArgumentParser):

    def __init__(self):
        super(Args, self).__init__()
        self.add_argument('-a', '--all', action='store_true')
        self.add_argument('-tn', '--triton', action='store_true')
        self.add_argument('-ts', '--torchserve', action='store_true')
        self.add_argument('-bt', '--bentoml', action='store_true')
        self.add_argument('-cl', '--cleanup', action='store_true')
        self._ns: argparse.Namespace = self.parse_args()
        self.all: bool = self._ns.all
        self.triton: bool = self._ns.triton
        self.torchserve: bool = self._ns.torchserve
        self.bentoml: bool = self._ns.bentoml
        self.cleanup: bool = self._ns.cleanup


class TestModel(ElectraModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pooling = torch.nn.Linear(256, 3)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        return self._pooling(output[0][:, 0, :])


class ModelBuilder:

    def __init__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'------------------- device : {str(device)} ------------------------')
        resource = "monologg/koelectra-small-v3-discriminator"
        self.tokenizer = ElectraTokenizer.from_pretrained(
            resource, clean_up_tokenization_spaces=False
        )

        self.model = (
            TestModel
            .from_pretrained(resource, torchscript=True)
            .to(device)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            example_text = "Replace me by any text you'd like."
            encoded_input = self.tokenizer(example_text, return_tensors='pt').to(device)
            self.traced_model = torch.jit.trace(
                self.model, [encoded_input['input_ids'], encoded_input['attention_mask']]
            )

    def save_model(self, path: Path):
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.jit.save(self.traced_model, str(path))
        print(f"save model     -> {path.absolute()}")

    def save_tokenizer(self, path: Path):
        path.parent.mkdir(exist_ok=True, parents=True)
        self.tokenizer.save_pretrained(str(path))
        print(f"save tokenizer -> {path.absolute()}")


def delete_recursively(path: Path):
    if path.is_dir():
        for p in path.glob('*'):
            delete_recursively(p)
        path.rmdir()
    else:
        print(f"delete : {path}")
        path.unlink()


if __name__ == '__main__':
    arg = Args()

    _triton = Path('triton/model_repository')
    _torchserve = Path('torchserve/ml')
    _bentoml = Path('bentoml/ml')

    if arg.cleanup:
        delete_recursively(_triton.joinpath('forward/1'))
        delete_recursively(_triton.joinpath('pre_processing/resource'))
        delete_recursively(_torchserve)
        delete_recursively(_bentoml)
    else:
        builder = ModelBuilder()
        if arg.triton or arg.all:
            builder.save_model(_triton.joinpath('forward/1/model.pt'))
            builder.save_tokenizer(_triton.joinpath('pre_processing/resource'))
        if arg.torchserve or arg.all:
            builder.save_model(_torchserve.joinpath('model/model.pt'))
            builder.save_tokenizer(_torchserve.joinpath('tokenizer'))
        if arg.bentoml or arg.all:
            builder.save_model(_bentoml.joinpath('model.pt'))
            builder.save_tokenizer(_bentoml.joinpath('tokenizer'))
