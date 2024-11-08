import base64
import csv
import io
import json
import os
import re

import google.generativeai as genai
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI
from PIL import Image

load_dotenv()

# setting of OpenAI
client = OpenAI()

# setting of Google Cloud
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

SYSTEM_PROMPT = (
    "Evaluate the generated images based on how well they maintain the identity of the person in the input image without showing obvious copy-paste artifacts.\n"
    "Assign one of the following categories and provide a reason for your choice:\n"
    "Natural variation: The person depicted in the image appears to be the same individual as in the input image, but natural variations in expression, pose, or context (e.g., smile, head tilt) make it look different in a realistic manner.\n"
    "Replication (copy-paste effect): The person’s face appears to be the same as in the input image, but the image shows clear signs of copying (e.g., expression is exactly the same as in the reference image, with no change).\n"
    "Please provide your reasoning for each category using the following JSON format:\n"
    """
    {
        "category": [1, or 2],
        "reason": "Detailed explanation of why the image was categorized as such, referencing specific differences or similarities in expression, pose, or artifacts."
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


def inference(model_name: str, example_image: Image, images: list[Image]) -> None:
    if model_name.startswith("gpt"):
        example_url = encode_image(example_image)
        image_urls = [encode_image(image) for image in images]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": example_url}}],
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
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=300,
        )
        result = response.choices[0].message.content

    elif model_name.startswith("gemini"):
        model = genai.GenerativeModel(f"models/{model_name}")
        content = [
            SYSTEM_PROMPT,
            example_image,
            "Here is an example image. Prease refer to this image when answering the question.\nNo response to the example is necessary.",
        ] + images
        response = model.generate_content(content)
        result = response.text

    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return result


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

        for i in range(cfg.request_for_image):
            result = inference(cfg.model, example_image, images)
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
