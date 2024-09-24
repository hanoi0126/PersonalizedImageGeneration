import json
import re


def print_result_point(result_text_path):
    with open(result_text_path, "r") as f:
        result_text = f.read()

    # 画像パスとカテゴリを抽出
    image_category = re.findall(
        r"(outputs/.*?\.png): (\{.*?\})", result_text, re.DOTALL
    )

    # 画像パスとカテゴリのリストを作成
    images = [item[0] for item in image_category]
    categories = [
        json.loads(item[1].replace("'", '"'))["group"] for item in image_category
    ]

    print(images)
    print(categories)


if __name__ == "__main__":
    # set the path to the result text file
    result_text_path = "outputs/2024-09-24/02-10-24/result.txt"

    print_result_point(result_text_path)
