import base64
import io
import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI
from PIL import Image

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = (
    "Do these images look like the person in the input image pasted on top?\n"
    "Please categorize each image into one of the following three groups:\n"
    "1. The person appears to be the same, but the image does not look pasted (natural differences in facial expression, orientation, etc.)\n"
    "2. The person appears to be the same, but the image looks pasted (identical facial expression, highly similar orientation)\n"
    "3. The person does not appear to be the same\n"
    "Provide your reasoning for the categorization.\n"
    "Output the results in the following json format and do not output any string other than json:\n"
    """
    {
        'group': 1,
        'reason': 'The same person is depicted, but the facial expression and the direction of the face are so different that it cannot be called a paste.'
    }
    """
)


def encode_image(image: Image) -> str:
    if image.mode == "RGBA":
        image = image.convert("RGB")  # RGBAからRGBに変換
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="JPEG")
    base64_image = (
        f"data:image/jpeg;base64,{base64.b64encode(byte_arr.getvalue()).decode()}"
    )
    return base64_image


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def generate_response_with_images(cfg: DictConfig) -> None:
    output_text = ""

    if cfg.generated_image_dir is not None:
        generated_image_paths = sorted(
            os.path.join(cfg.generated_image_dir, f)
            for f in os.listdir(cfg.generated_image_dir)
            if f.endswith(".png") and not f.startswith("grid")
        )
    else:
        generated_image_paths = [cfg.generated_image]

    for idx, generated_image_path in enumerate(generated_image_paths):
        print(f"Processing {generated_image_path}...")
        example_image = Image.open(cfg.few_shot_example)
        images = [
            Image.open(image_path)
            for image_path in [cfg.reference_image, generated_image_path]
        ]
        example_url = encode_image(example_image)
        image_urls = [encode_image(image) for image in images]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": example_url}}
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Here is an example image. This corresponds to 5.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}}
                        for image_url in image_urls
                    ],
                },
            ],
            max_tokens=300,
        )

        result = response.choices[0].message.content
        print(result)

        output_text += f"{generated_image_path}: {result}\n"

    with open(f"{cfg.output_dir}/result.txt", "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    print("========== system prompt ==========")
    print(SYSTEM_PROMPT)
    print("===================================")
    generate_response_with_images()
