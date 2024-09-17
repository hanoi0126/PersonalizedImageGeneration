import cv2
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="eval_config")
def template_matching(cfg: DictConfig) -> None:

    reference_image = cv2.imread(cfg.reference_image)
    genearted_image = cv2.imread(cfg.generated_image)
    w, h = reference_image.shape[::-1]

    result = cv2.matchTemplate(genearted_image, reference_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(genearted_image, top_left, bottom_right, 255, 2)
    cv2.imwrite(f"{cfg.output_dir}/output_0.png", genearted_image)


if __name__ == "__main__":
    template_matching()
