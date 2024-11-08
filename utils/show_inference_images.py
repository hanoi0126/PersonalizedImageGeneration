import os

import matplotlib.pyplot as plt


def save_grid_image(image_dir, ncol=4, vlm_result=None, landmark_result=None) -> None:
    images = [
        (i, plt.imread(f"{image_dir}/{i}"))
        for i in sorted(os.listdir(image_dir))
        if (i.endswith(".png") or i.endswith("jpg")) and not i.startswith("grid")
    ]

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
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(image)
        plt.title(label, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{image_dir}/grid.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    # set image_dir to the directory containing the images
    image_dir = "outputs/2024-10-30/12-53-42"

    vlm_result = None
    landmark_result = None

    save_grid_image(image_dir, vlm_result=vlm_result, landmark_result=landmark_result)
