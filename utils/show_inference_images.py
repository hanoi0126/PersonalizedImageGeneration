import os

import matplotlib.pyplot as plt


def save_grid_image(image_dir, nrow=4):
    images = [
        plt.imread(f"{image_dir}/{i}")
        for i in os.listdir(image_dir)
        if i.endswith(".png")
    ]

    plt.figure(figsize=(20, 20))
    for i, image in enumerate(images):
        plt.subplot(nrow, nrow, i + 1)
        plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{image_dir}/grid.png")
    plt.close()


if __name__ == "__main__":

    # set image_dir to the directory containing the images
    image_dir = "outputs/2024-09-17/18-52-50"

    save_grid_image(image_dir)
