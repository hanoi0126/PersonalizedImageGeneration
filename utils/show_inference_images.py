import os

import matplotlib.pyplot as plt


def save_grid_image(image_dir, nrow=4, vlm_result=None):
    images = [
        (i, plt.imread(f"{image_dir}/{i}"))
        for i in os.listdir(image_dir)
        if i.endswith(".png")
    ]

    plt.figure(figsize=(20, 20))
    for i, (filename, image) in enumerate(images):
        label = filename if vlm_result is None else f"{filename} : {vlm_result[i-1]}"
        plt.subplot(nrow, nrow, i + 1)
        plt.imshow(image)
        plt.title(label)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{image_dir}/grid.png")
    plt.close()


if __name__ == "__main__":

    # set image_dir to the directory containing the images
    image_dir = "outputs/2024-09-24/02-06-39"
    vlm_result = [2, 1, 3, 3, 3, 1, 3, 1, 3, 3, 1]

    save_grid_image(image_dir, vlm_result=vlm_result)
