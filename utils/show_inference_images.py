import os

import matplotlib.pyplot as plt


def save_grid_image(image_dir, ncol=4, vlm_result=None, landmark_result=None) -> None:
    images = [
        (i, plt.imread(f"{image_dir}/{i}"))
        for i in os.listdir(image_dir)
        if i.endswith(".png") and not i.startswith("grid")
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
    image_dir = "outputs/2024-09-24/21-49-47"

    # set vlm_result to the list of categories predicted by the model
    # vlm_result_csv = "outputs/2024-09-24/14-32-05/result.csv"
    # result_df = pd.read_csv(vlm_result_csv)
    # vlm_result = result_df["category"].tolist()
    vlm_result = None

    # set landmark result to the list of landmarks predicted by the model
    # landmark_result_csv = "outputs/2024-09-24/15-11-42/result.csv"
    # landmark_result_df = pd.read_csv(landmark_result_csv)
    # landmark_result = landmark_result_df["difference"].tolist()
    landmark_result = None

    save_grid_image(image_dir, vlm_result=vlm_result, landmark_result=landmark_result)
