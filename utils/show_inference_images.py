import os

import matplotlib.pyplot as plt
import pandas as pd


def save_grid_image(
    image_dir, ncol=4, vlm_result=None, landmark_result=None, sort_with_score=None
) -> None:
    images = [
        (i, plt.imread(f"{image_dir}/{i}"))
        for i in sorted(os.listdir(image_dir))
        if (i.endswith(".png") or i.endswith("jpg")) and not i.startswith("grid")
    ]

    if sort_with_score is not None:
        df = pd.read_csv(sort_with_score)
        images_and_score = []
        for filename, _ in images:
            filepath = os.path.join(image_dir, filename)
            clip_i = df[df["image_path"] == filepath]["sim_img2img"].values[0]
            clip_t = df[df["image_path"] == filepath]["sim_txt2img"].values[0]
            score = df[df["image_path"] == filepath]["face_separation"].values[0]
            images_and_score.append(
                (filename, plt.imread(filepath), score, clip_i, clip_t)
            )
        images_and_score = sorted(images_and_score, key=lambda x: x[2], reverse=True)
        images = [(filename, image) for filename, image, _, _, _ in images_and_score]

    num_images = len(images)
    nrow = (num_images + ncol - 1) // ncol

    fig_width = min(20, 5 * ncol)
    fig_height = min(50, 5 * nrow)

    plt.figure(figsize=(fig_width, fig_height))
    for i, (filename, image) in enumerate(images):
        label = filename if vlm_result is None else f"{filename} : {vlm_result[i]}"
        if (
            landmark_result is not None
            and filename.endswith("_white.png")
            and not filename.startswith("reference")
        ):
            label += f"\n{landmark_result[i // 2 - 1]:.2f}"
        if sort_with_score is not None:
            label += f"\nfs: {images_and_score[i][2]}\nci: {images_and_score[i][3]}\nct: {images_and_score[i][4]}"
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(image)
        plt.title(label, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{image_dir}/grid.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    # set image_dir to the directory containing the images
    image_dir = "data/celeba_inpaint_2"

    vlm_result = None
    landmark_result = None
    sort_with_score = None

    save_grid_image(
        image_dir,
        vlm_result=vlm_result,
        landmark_result=landmark_result,
        sort_with_score=sort_with_score,
    )
