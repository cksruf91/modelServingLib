import importlib
import os
import sys
from typing import Dict

from locust import between

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
hopper = importlib.import_module("grasshopper")


class TritonHopper(hopper.Grasshopper):
    wait_time = between(0.1, 0.4)
    embedding_endpoint = "/v2/models/cls/versions/1/infer"

    @staticmethod
    def _get_body(text: str) -> Dict:
        data = [
            [text],
        ]
        shape = [len(data), len(data[0])]
        return {
            "name": "cls",
            "inputs": [
                {
                    "name": "input_text",
                    "shape": shape,
                    "datatype": "BYTES",
                    "data": data
                }
            ]
        }
