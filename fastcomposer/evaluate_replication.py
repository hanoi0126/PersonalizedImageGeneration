import base64
import io

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = (
    "Do these images look like the person in the input image pasted on top?"
    "On a scale of 1 to 5, how would you rate them?"
    "The closer to 5, the more the image looks pasted."
)


def encode_image(image: Image) -> str:
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="JPEG")
    base64_image = (
        f"data:image/jpeg;base64,{base64.b64encode(byte_arr.getvalue()).decode()}"
    )
    return base64_image


def generate_response_with_images(images: list[Image]) -> str:
    image_urls = [encode_image(image) for image in images]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
    return response.choices[0].message.content


if __name__ == "__main__":
    input_image = Image.open("data/sample/einstein.png")
    output_image = Image.open("outputs/sample/output_0.png")
    response = generate_response_with_images([input_image, output_image])
    print(response)
