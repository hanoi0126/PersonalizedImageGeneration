import os

import cv2
import hydra
import numpy as np
from facenet_pytorch import MTCNN
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor


class FaceDetector:
    def __init__(self):
        self.mtcnn = MTCNN(margin=0, keep_all=True)

    def get_face_deteil(self, image: Image):
        if image.mode != "RGB":
            image = image.convert("RGB")

        face_bboxes, probabilities = self.mtcnn.detect(image)
        if face_bboxes is None or len(face_bboxes) == 0:
            raise ValueError("顔が検出されませんでした。")

        return face_bboxes, probabilities

    def to_numpy(self, image: Tensor) -> np.ndarray:
        image = (
            image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        )  # (C, H, W) -> (H, W, C)
        image = (image * 255).astype(np.uint8)  # [0, 1] -> [0, 255]

        return image

    def to_pil_image(self, image: Tensor) -> Image:
        image = (
            image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        )  # (C, H, W) -> (H, W, C)
        image = (image * 255).astype(np.uint8)  # [0, 1] -> [0, 255]

        return Image.fromarray(image)  # PIL.Image に変換


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def template_matching(cfg: DictConfig) -> None:
    output_text = ""

    detector = FaceDetector()
    reference_image = Image.open(cfg.reference_image)
    face_bboxes, probabilities = detector.get_face_deteil(reference_image)

    template_image = reference_image.crop(face_bboxes[0].tolist())
    h, w = template_image.size
    template_image = template_image.resize((h // 4, w // 4))

    template_image.save(f"{cfg.output_dir}/detected_face.png")

    template_image = cv2.imread(f"{cfg.output_dir}/detected_face.png")

    if cfg.generated_image_dir is not None:
        generated_image_paths = sorted(
            os.path.join(cfg.generated_image_dir, f)
            for f in os.listdir(cfg.generated_image_dir)
            if f.endswith(".png")
        )
    else:
        generated_image_paths = [cfg.generated_image]

    for idx, generated_image_path in enumerate(generated_image_paths):
        print(f"Processing {generated_image_path}...")
        generated_image = cv2.imread(generated_image_path)

        if generated_image is None:
            raise ValueError("生成画像が正しく読み込まれませんでした。")

        h, w = template_image.shape[:2]

        result = cv2.matchTemplate(
            generated_image, template_image, cv2.TM_CCOEFF_NORMED
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print("Max value:", max_val, "Max location:", max_loc)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(
            generated_image, top_left, bottom_right, thickness=4, color=(0, 255, 255)
        )

        cv2.imwrite(f"{cfg.output_dir}/result_{idx:02}.png", generated_image)

        output_text += f"{generated_image_path}: {max_val}\n"

    with open(f"{cfg.output_dir}/result.txt", "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    template_matching()
