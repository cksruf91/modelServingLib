import string
from typing import List
import numpy as np

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def __init__(self):
        self.logger = pb_utils.Logger
        self.label = {i: s for i, s in enumerate(string.ascii_uppercase)}

    def initialize(self, args):
        ...

    def _label_mapping(self, k: int) -> bytes:
        return self.label.get(k).encode('utf-8', 'ignore')

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """ PostProcessing Step
        1. ordering output
        2. label mapping
        """
        responses = []
        for request in requests:
            model_output = (
                pb_utils.get_input_tensor_by_name(request, "output__0")
                .as_numpy()
            )
            self.logger.log_info(f"model_output: {model_output.shape}")
            indices = np.argsort(model_output, axis=1)
            classes = np.vectorize(self._label_mapping)(indices)  # index -> label mapping
            values = np.sort(model_output, axis=1)

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor('class', classes),
                pb_utils.Tensor('prob', values)
            ])
            responses.append(inference_response)

        return responses

    def finalize(self):
        ...
