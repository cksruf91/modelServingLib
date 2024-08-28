from typing import Dict

from locust import HttpUser, task, between


class LocustUser(HttpUser):
    wait_time = between(0.1, 0.4)
    embedding_endpoint = "/predictions/cls"
    abstract = True

    @staticmethod
    def _get_body(text: str) -> Dict:
        return {
            "text": text
        }

    def on_start(self):
        self.client.post(
            self.embedding_endpoint,
            json=self._get_body('model warm up text')
        )


class User1(LocustUser):

    @task(weight=1)
    def cls_request(self):
        text = "슈퍼 엔저 장기화에…한국 수출∙경상수지에 비상등"
        body = self._get_body(text)
        with self.client.post(self.embedding_endpoint, json=body, catch_response=True) as response:
            if response.elapsed.total_seconds() > 1.5:
                response.failure("Request took too long")


class User2(LocustUser):
    @task(weight=1)
    def cls_request(self):
        text = "Kamala Harris is more trusted than Donald Trump on the US economy"
        body = self._get_body(text)
        with self.client.post(self.embedding_endpoint, json=body, catch_response=True) as response:
            if response.elapsed.total_seconds() > 1.5:
                response.failure("Request took too long")


class User3(LocustUser):
    @task(weight=1)
    def cls_request(self):
        text = ("The fact that voters were more positive on Harris than on Biden... "
                "says as much about how badly Biden was doing as it does about how well Harris is doing,")
        body = self._get_body(text)
        with self.client.post(self.embedding_endpoint, json=body, catch_response=True) as response:
            if response.elapsed.total_seconds() > 1.5:
                response.failure("Request took too long")
