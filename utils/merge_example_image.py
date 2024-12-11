import matplotlib.pyplot as plt
from PIL import Image


# def merge_images(input_path, good_output_path, replicate_output_path, bad_output_path, output_path):
#     # 画像を読み込み、リサイズする
#     images = {
#         "Input": Image.open(input_path),
#         "1 : Good Output": Image.open(good_output_path),
#         "2 : Replicate Output": Image.open(replicate_output_path),
#         "3 : Bad Output": Image.open(bad_output_path),
#     }
#     for key in images:
#         images[key] = images[key].resize((256, 256))

#     # 2x2のサブプロットを作成
#     fig, axs = plt.subplots(2, 2, figsize=(12, 12))
#     fig.suptitle("Image Comparison", fontsize=16)

#     # 各サブプロットに画像を配置
#     for ax, (title, img) in zip(axs.flatten(), images.items()):
#         ax.imshow(img)
#         ax.axis("off")
#         ax.set_title(title)

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.close()


def merge_images(input_path, good_output_path, replicate_output_path, output_path):
    # 画像を読み込み、リサイズする
    images = {
        "Reference Image (Input)": Image.open(input_path),
        "Good Output (Not Replication)": Image.open(good_output_path),
        "Bad Output (Replication)": Image.open(replicate_output_path),
    }
    for key in images:
        images[key] = images[key].resize((256, 256))

    # 横に3枚並べる
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Image Comparison", fontsize=16)

    # 各画像をプロット
    for ax, (title, img) in zip(axs, images.items()):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)

    # レイアウト調整と保存
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    merge_images(
        input_path="data/example/sample_input.png",
        good_output_path="data/example/sample_good_output.png",
        replicate_output_path="data/example/sample_replicate_output.png",
        output_path="data/example/merged_image_modified.png",
    )
