import cv2
import numpy as np
import tritonclient.http as httpclient


def get_image(path) -> np.ndarray:
    image = cv2.imread(path)
    image.resize(224, 224, 3)  # required input image size
    image = image.transpose(2, 0, 1)
    # normalize
    return (image / image.mean()).astype(np.float32)


class TritonHttpClient(httpclient.InferenceServerClient):
    def __call__(self, image: np.ndarray):
        inputs = httpclient.InferInput("input__0", list(image.shape), datatype="FP32")
        inputs.set_data_from_numpy(image, binary_data=True)

        outputs = httpclient.InferRequestedOutput("output__0", binary_data=True, class_count=1000)

        result = self.infer(model_name='resnet50', inputs=[inputs], outputs=[outputs])
        return result.as_numpy('output__0')


if __name__ == '__main__':
    img1 = get_image("./img1.jpg")
    print(f"img1 : {img1.shape}")
    client = TritonHttpClient(url="localhost:8000")
    predictions = client(img1)
    print(f"predictions : {predictions[:10]}")
