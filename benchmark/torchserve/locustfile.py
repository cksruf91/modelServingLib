import importlib
import os
import sys
from typing import Dict

from locust import between

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
hopper = importlib.import_module("grasshopper")


class TorchServeLocust(hopper.Grasshopper):
    wait_time = between(0.1, 0.4)
    embedding_endpoint = "/predictions/cls"

    @staticmethod
    def _get_body(text: str) -> Dict:
        return {
            "text": text
        }
