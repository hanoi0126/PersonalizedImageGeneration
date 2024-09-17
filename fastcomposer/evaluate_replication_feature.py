import cv2
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def feature_matching(cfg: DictConfig) -> None:

    reference_image = cv2.imread(cfg.reference_image)
    generated_image = cv2.imread(cfg.generated_image)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(reference_image, None)
    kp2, des2 = orb.detectAndCompute(generated_image, None)

    # BFMatcherによる特徴点のマッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # マッチング結果の描画
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(
        reference_image, kp1, generated_image, kp2, matches[:10], None, flags=2
    )

    cv2.imwrite(f"{cfg.output_dir}/result.png", img_matches)


if __name__ == "__main__":
    feature_matching()
