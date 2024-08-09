from typing import List, Dict
import numpy as np
from pathlib import Path

from transformers import DistilBertTokenizer, TensorType
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def __init__(self):
        self.tokenizer_path = Path("/models/tokenizer/resource")
        self.tokenizer = DistilBertTokenizer.from_pretrained(str(self.tokenizer_path))
        self.logger = pb_utils.Logger

    def initialize(self, args):
        self.logger.log_info(f'tokenizer from {self.tokenizer_path.absolute()}')

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "input_text")
                .as_numpy()
                .tolist()
            ]
            self.logger.log_info(str(query))
            tokens: Dict[str, np.ndarray] = self.tokenizer(
                text=query, return_tensors=TensorType.NUMPY, padding=True
            )
            # tensorrt uses int32 as input type, ort uses int64
            tokens = {k: v.astype(np.int64) for k, v in tokens.items()}
            # communicate the tokenization results to Triton server
            outputs = list()
            for input_name in self.tokenizer.model_input_names:
                tensor_input = pb_utils.Tensor(input_name, tokens[input_name])
                outputs.append(tensor_input)

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses

    def finalize(self):
        ...
