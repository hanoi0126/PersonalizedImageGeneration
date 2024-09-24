import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    ex_input = Image.open("data/example/sample_input.png")
    ex_output = Image.open("data/example/sample_output.png")
    ex_input = ex_input.resize((256, 256))
    ex_output = ex_output.resize((256, 256))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ex_input)
    plt.axis("off")
    plt.title("Input Image")

    plt.subplot(1, 2, 2)
    plt.imshow(ex_output)
    plt.axis("off")
    plt.title("Output Image")

    plt.tight_layout()
    plt.savefig("data/example/merged_image.png")
    plt.close()
