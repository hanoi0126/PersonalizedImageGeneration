import base64
import csv
import io
import json
import os
import re

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI
from PIL import Image

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = (
    "Do these images look like the person in the input image pasted on top?\n"
    "Please categorize each image into one of the following three categories:\n"
    "1. The person appears to be the same, but the image does not look pasted (natural differences in facial expression, orientation, etc.)\n"
    "2. The identical person is depicted, but only the face portion is in a paste-on format, showing an replication.\n"
    "3. The person does not appear to be the same\n"
    "Provide your reasoning for the categorization.\n"
    "Output the results in the following json format and do not output any string other than json:\n"
    """
    {
        'categgory': 1,
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
    output_dict = {}

    if cfg.generated_image_dir is not None:
        generated_image_paths = sorted(
            os.path.join(cfg.generated_image_dir, f)
            for f in os.listdir(cfg.generated_image_dir)
            if (f.endswith(".png") or f.endswith(".jpg")) and not f.startswith("grid")
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
        for i in range(cfg.request_for_image):
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
                        "content": "Here is an example image. Prease refer to this image when answering the question.",
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

            output_text += f"{generated_image_path}_{i:02}: {result}\n"
            match = re.search(r"{(.|\s)*}", result.replace("'", '"'))

            if match:
                # JSONとして扱うために辞書に変換
                try:
                    json_data = json.loads(match.group())
                    output_dict[f"{generated_image_path}_{i:02}"] = json_data
                except json.JSONDecodeError:
                    print("JSON decode error")
                    continue


    with open(f"{cfg.output_dir}/result.txt", "w") as f:
        f.write(output_text)

    # CSVファイルに書き込む
    with open(f"{cfg.output_dir}/result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)

        # ヘッダーを書く (ファイルパスとJSONのキーを含める)
        writer.writerow(["generated_image_path", "category", "reason"])

        # 辞書の内容を書き込む
        for image_path, json_data in output_dict.items():
            # 辞書の各キーの値を順に取り出して書き込む
            writer.writerow([image_path, json_data["category"], json_data["reason"]])


if __name__ == "__main__":
    print("========== system prompt ==========")
    print(SYSTEM_PROMPT)
    print("===================================")
    generate_response_with_images()
