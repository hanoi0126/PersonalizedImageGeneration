import os

import pandas as pd
import matplotlib.pyplot as plt


def save_grid_image(image_dir, ncol=4, vlm_result=None):
    images = [
        (i, plt.imread(f"{image_dir}/{i}"))
        for i in os.listdir(image_dir)
        if i.endswith(".png") and not i.startswith("grid")
    ]

    num_images = len(images)
    nrow = (num_images +  ncol - 1) // ncol 

    fig_width = min(20, 5 * ncol)  
    fig_height = min(50, 5 * nrow) 

    plt.figure(figsize=(fig_width, fig_height))
    for i, (filename, image) in enumerate(images):
        label = filename if vlm_result is None else f"{filename} : {vlm_result[i]}"
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(image)
        plt.title(label, fontsize=8)
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(f"{image_dir}/grid.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    # set image_dir to the directory containing the images
    image_dir = "outputs/2024-09-24/14-27-46"

    vlm_result_csv = "outputs/2024-09-24/14-32-05/result.csv"
    result_df = pd.read_csv(vlm_result_csv)
    vlm_result = result_df["category"].tolist()

    save_grid_image(image_dir, vlm_result=vlm_result)
