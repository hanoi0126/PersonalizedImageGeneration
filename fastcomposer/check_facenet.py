import os

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from icecream import ic
from PIL import Image
import time


class FaceNet:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mtcnn = MTCNN(margin=0, keep_all=True, device=self.device)
        self.embedding_model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        ic(self.device)

    def _detect_faces(self, image_pil: Image) -> torch.Tensor:
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        return self.mtcnn(image_pil)

    def _get_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        return self.embedding_model(image_tensor.unsqueeze(0).to(self.device)).detach().cpu().numpy().flatten()

    def _calc_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def calc_face_similarity(self, image1: Image, image2: Image) -> float:
        image1_tensor = self._detect_faces(image1)[0]
        image2_tensor = self._detect_faces(image2)[0]
        embedding1 = self._get_embedding(image1_tensor)
        embedding2 = self._get_embedding(image2_tensor)

        return self._calc_similarity(embedding1, embedding2)

    def calc_face_essential_similarity(self, image1: Image, image2: Image) -> float:
        # rotate image
        stack_embedding_image1 = []
        stack_embedding_image2 = []
        for angle in range(0, 360, 10):
            times = [time.time()]
            image1_rotated = image1.rotate(angle)
            image2_rotated = image2.rotate(angle)
            times.append(time.time())
            # print(f"rotate time: {times[-1] - times[-2]}")
            image1_tensor = self._detect_faces(image1_rotated)
            image2_tensor = self._detect_faces(image2_rotated)
            times.append(time.time())
            # print(f"detect time: {times[-1] - times[-2]}")
            if image1_tensor is None or image2_tensor is None:
                continue
            embedding1 = self._get_embedding(image1_tensor[0])
            embedding2 = self._get_embedding(image2_tensor[0])
            times.append(time.time())
            # print(f"embedding time: {times[-1] - times[-2]}")
            stack_embedding_image1.append(embedding1)
            stack_embedding_image2.append(embedding2)

        if not stack_embedding_image1 or not stack_embedding_image2:
            raise ValueError("No embeddings could be calculated from the rotated images.")

        # average
        mean_embedding_image1 = np.mean(stack_embedding_image1, axis=0)
        mean_embedding_image2 = np.mean(stack_embedding_image2, axis=0)

        return self._calc_similarity(mean_embedding_image1, mean_embedding_image2)


if __name__ == "__main__":
    facenet = FaceNet()
    # image_pil = Image.open("data/celeba/000023/000023_hb_x4_resize.png")
    # for angle in range(0, 91, 10):
    #     image_pil_rotate = image_pil.rotate(angle)
    #     image_tensor = facenet._detect_faces(image_pil_rotate)
    #     embedding = facenet._get_embedding(image_tensor[0])

    #     ic(angle, type(image_tensor), image_tensor.shape, type(embedding), embedding.shape)

    # image1 = Image.open("data/celeba/067855_hb_x4_resize.png")
    # image2 = Image.open("data/celeba/142586_hb_x4_resize.png")

    image1 = Image.open("data/example/sample_input.png")
    image2 = Image.open("data/example/sample_good_output.png")

    simple_similarity = facenet.calc_face_similarity(image1, image2)
    stack_similarity = facenet.calc_face_essential_similarity(image1, image2)
    ic(simple_similarity, stack_similarity, stack_similarity - simple_similarity)

    # image_dir = "data/celeba/00001"
    # image_dir = "data/celeba_inpaint_2"

    # for image_file in sorted(os.listdir(image_dir)[:10]):
    #     image2 = Image.open(os.path.join(image_dir, image_file))

    #     simple_similarity = facenet.calc_face_similarity(image1, image2)
    #     stack_similarity = facenet.calc_face_essential_similarity(image1, image2)
    #     ic(simple_similarity, stack_similarity, stack_similarity - simple_similarity)
