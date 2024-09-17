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
            if f.endswith(".png") and not f.startswith("grid")
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


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def template_matching_for_diff_size(cfg: DictConfig) -> None:
    output_text = ""

    print("Loading FaceDetector...")
    detector = FaceDetector()
    reference_image = Image.open(cfg.reference_image)
    face_bboxes, probabilities = detector.get_face_deteil(reference_image)

    template_image = reference_image.crop(face_bboxes[0].tolist())
    h, w = template_image.size
    template_image = template_image.resize((h // 4, w // 4))
    h, w = template_image.size

    template_image.save(f"{cfg.output_dir}/detected_face.png")

    template_image = cv2.imread(f"{cfg.output_dir}/detected_face.png")

    if cfg.generated_image_dir is not None:
        generated_image_paths = sorted(
            os.path.join(cfg.generated_image_dir, f)
            for f in os.listdir(cfg.generated_image_dir)
            if f.endswith(".png") and not f.startswith("grid")
        )
    else:
        generated_image_paths = [cfg.generated_image]

    for idx, generated_image_path in enumerate(generated_image_paths):
        print(f"Processing {generated_image_path}...")
        generated_image = cv2.imread(generated_image_path)

        if generated_image is None:
            raise ValueError("生成画像が正しく読み込まれませんでした。")

        result_image = generated_image.copy()
        best_val = 0
        loc = (0, 0)
        g_h, g_w, _ = generated_image.shape
        max_expansion_rate = min(g_h // h, g_w // w) * 100
        print(f"Expansion rate: {max_expansion_rate}")

        for i in range(50, max_expansion_rate, 5):
            expansion_rate = i / 100
            t2 = template_image.copy()
            t3 = cv2.resize(
                t2,
                None,
                fx=expansion_rate,
                fy=expansion_rate,
                interpolation=cv2.INTER_CUBIC,
            )

            result = cv2.matchTemplate(generated_image, t3, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = max_val
                loc = max_loc
                t_h, t_w, _ = t3.shape


        result_image = cv2.rectangle(
            result_image,
            loc,
            (loc[0] + t_w, loc[1] + t_h),
            thickness=4,
            color=(0, 255, 255),
        )

        cv2.imwrite(f"{cfg.output_dir}/result_{idx:02}.png", result_image)
        output_text += f"{generated_image_path}: {best_val}\n"

    with open(f"{cfg.output_dir}/result.txt", "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    # template_matching()
    template_matching_for_diff_size()
