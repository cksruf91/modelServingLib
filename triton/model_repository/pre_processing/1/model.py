from pathlib import Path
from typing import List, Dict

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, TensorType


class TritonPythonModel:
    def __init__(self):
        self.tokenizer_path = Path("/models/pre_processing/resource")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
        self.logger = pb_utils.Logger

    def initialize(self, args):
        self.logger.log_info(f'tokenizer from {self.tokenizer_path.absolute()}')

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """ PreProcessing Step
        1. tokenizing text
        """
        responses = []
        # for loop for batch requests
        for request in requests:
            query = [
                t[0].decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "input_text")
                .as_numpy()
                .tolist()
            ]
            self.logger.log_info(f"input text : {str(query)}")
            tokens: Dict[str, np.ndarray] = self.tokenizer(
                text=query, return_tensors=TensorType.NUMPY, padding='max_length', truncation=True, max_length=100
            )
            tokens = {k: v.astype(np.int64) for k, v in tokens.items()}
            outputs = list()
            for input_name in self.tokenizer.model_input_names:
                self.logger.log_info(f"token size : {input_name} -> {tokens[input_name].shape}")
                tensor_input = pb_utils.Tensor(input_name, tokens[input_name])
                outputs.append(tensor_input)

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses

    def finalize(self):
        ...
