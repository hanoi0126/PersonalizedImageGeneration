import base64
import csv
import io
import json
import os
import pickle
import re

import google.generativeai as genai
import hydra
from dotenv import load_dotenv
from icecream import ic
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


def json2dict(json_string, error_key="error") -> dict:

    def _extract_string(text, start_string=None, end_string=None) -> str:
        # 最初の文字
        if start_string is not None and start_string in text:
            idx_head = text.index(start_string)
            text = text[idx_head:]
        # 最後の文字
        if end_string is not None and end_string in text:
            idx_tail = len(text) - text[::-1].index(end_string[::-1])
            text = text[:idx_tail]
        return text

    try:
        python_dict = json.loads(
            _extract_string(json_string, start_string="{", end_string="}"),
            strict=False,
        )
    except ValueError:
        if error_key is None:
            return json_string
        python_dict = {error_key: json_string}
    if isinstance(python_dict, dict):
        return python_dict
    return {error_key: python_dict}


def encode_image(image: Image) -> str:
    if image.mode == "RGBA":
        image = image.convert("RGB")  # RGBAからRGBに変換
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="JPEG")
    base64_image = f"data:image/jpeg;base64,{base64.b64encode(byte_arr.getvalue()).decode()}"
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
                "content": [{"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls],
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

    if cfg.reference_dir is not None:
        iteration = len(sorted(os.listdir(cfg.reference_dir)))
        reference_image_dirs = [os.path.join(cfg.reference_dir, f) for f in sorted(os.listdir(cfg.reference_dir))]
        reference_images = [
            os.path.join(reference_image_dir, os.listdir(reference_image_dir)[0])
            for reference_image_dir in reference_image_dirs
        ]
    else:
        iteration = 1
        reference_images = [cfg.reference_image]

    ic(len(reference_image_dirs))
    ic(len(generated_image_paths))
    ic(len(cfg.caption_list))
    assert len(reference_image_dirs) == len(generated_image_paths) // len(cfg.caption_list)

    for iter_idx, reference_image in enumerate(reference_images):
        start = iter_idx * len(cfg.caption_list)
        end = (iter_idx + 1) * len(cfg.caption_list) - 15
        for idx, generated_image_path in enumerate(generated_image_paths[start:end]):
            # print(f"Processing {generated_image_path}...")
            ic(idx, reference_image, generated_image_path)
            example_image = Image.open(cfg.few_shot_example)
            images = [Image.open(image_path) for image_path in [reference_image, generated_image_path]]

            for i in range(cfg.request_for_image):
                result = inference(cfg.model, example_image, images)

                output_text += f"{generated_image_path}_{i:02}: {result}\n"
                result_dict = json2dict(result)
                output_dict[f"{generated_image_path}_{i:02}"] = result_dict
                ic(result_dict)

    with open(f"{cfg.output_dir}/result.txt", "w") as f:
        f.write(output_text)

    # pklファイルに書き込む
    with open(f"{cfg.output_dir}/result.pkl", "wb") as f:
        pickle.dump(output_dict, f)

    # CSVファイルに書き込む
    with open(f"{cfg.output_dir}/result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)

        # ヘッダーを書く (ファイルパスとJSONのキーを含める)
        writer.writerow(["generated_image_path", "category", "reason"])

        # 辞書の内容を書き込む
        for image_path, dict_data in output_dict.items():
            # 辞書の各キーの値を順に取り出して書き込む
            writer.writerow([image_path, dict_data["category"], dict_data["reason"]])


if __name__ == "__main__":
    print("========== system prompt ==========")
    print(SYSTEM_PROMPT)
    print("===================================")
    generate_response_with_images()
