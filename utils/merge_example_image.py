import matplotlib.pyplot as plt
from PIL import Image


def merge_images(
    input_path, good_output_path, replicate_output_path, bad_output_path, output_path
):
    # 画像を読み込み、リサイズする
    images = {
        "Input": Image.open(input_path),
        "1 : Good Output": Image.open(good_output_path),
        "2 : Replicate Output": Image.open(replicate_output_path),
        "3 : Bad Output": Image.open(bad_output_path),
    }
    for key in images:
        images[key] = images[key].resize((256, 256))

    # 2x2のサブプロットを作成
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Image Comparison", fontsize=16)

    # 各サブプロットに画像を配置
    for ax, (title, img) in zip(axs.flatten(), images.items()):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    merge_images(
        input_path="data/example/sample_input.png",
        good_output_path="data/example/sample_good_output.png",
        replicate_output_path="data/example/sample_replicate_output.png",
        bad_output_path="data/example/sample_bad_output.png",
        output_path="data/example/merged_image.png",
    )
