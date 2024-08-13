from locust import HttpUser, task, between


class EmbeddingUser(HttpUser):
    wait_time = between(0.1, 0.4)
    embedding_endpoint = "/v2/models/embedding/versions/1/infer"
    abstract = True

    @staticmethod
    def _get_body(text):
        data = [
            [text],
        ]
        shape = [len(data), len(data[0])]
        body = {
            "name": "embedding",
            "inputs": [
                {
                    "name": "input_text",
                    "shape": shape,
                    "datatype": "BYTES",
                    "data": data
                }
            ]
        }
        return body


class User1(EmbeddingUser):
    @task(weight=1)
    def embedding(self):
        text = "슈퍼 엔저 장기화에…한국 수출∙경상수지에 비상등"
        body = self._get_body(text)
        self.client.post(self.embedding_endpoint, json=body)


class User2(EmbeddingUser):
    @task(weight=1)
    def embedding(self):
        text = "Kamala Harris is more trusted than Donald Trump on the US economy"
        body = self._get_body(text)
        self.client.post(self.embedding_endpoint, json=body)


class User3(EmbeddingUser):
    @task(weight=1)
    def embedding(self):
        text = ("The fact that voters were more positive on Harris than on Biden... "
                "says as much about how badly Biden was doing as it does about how well Harris is doing,")
        body = self._get_body(text)
        self.client.post(self.embedding_endpoint, json=body)
