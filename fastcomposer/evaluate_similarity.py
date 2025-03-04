import os

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from icecream import ic
from omegaconf import DictConfig
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class Evaluator:
    def __init__(self, ref_image_path):
        # デバイス設定（GPUが利用可能な場合にはGPUに設定）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.mtcnn = MTCNN(margin=0, keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(
            self.device
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        self.ref_image = Image.open(ref_image_path)
        self.ref_boxes, _, self.ref_points = self.mtcnn.detect(
            Image.open(ref_image_path), landmarks=True
        )
        self.ref_face = self.get_subject_from_image(Image.open(ref_image_path))

    def encode_text(self, text):
        inputs = self.clip_processor(text=text, return_tensors="pt").to(self.device)
        text_features = self.clip_model.get_text_features(**inputs)
        return text_features

    def encode_image(self, img):
        inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
        image_features = self.clip_model.get_image_features(**inputs)
        return image_features

    def calc_sim_txt2img(self, text, img):
        text_embedding = self.encode_text(text)
        img_embedding = self.encode_image(img)
        return self.cosine_similarity(text_embedding, img_embedding)

    def cosine_similarity(self, tensor_1, tensor_2):
        tensor_1 = tensor_1.to(self.device)
        tensor_2 = tensor_2.to(self.device)

        if tensor_1.dim() == 1:
            tensor_1 = tensor_1.unsqueeze(0)
        if tensor_2.dim() == 1:
            tensor_2 = tensor_2.unsqueeze(0)

        dot = torch.matmul(tensor_1, tensor_2.T)
        norm_tensor_1 = torch.norm(tensor_1, dim=1)
        norm_tensor_2 = torch.norm(tensor_2, dim=1)
        cosine = dot / (norm_tensor_1 * norm_tensor_2)
        return cosine.mean().item()

    def get_subject_from_image(self, image, landmarks=False):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.mtcnn(image)

    def calc_sim_img2img(self, gen_image):
        gen_subjects = self.get_subject_from_image(gen_image)
        if gen_subjects is None:
            print("No face detected in the generated image")
            return 0.0

        ref_embedding = self.resnet(self.ref_face[0].to(self.device).unsqueeze(0))
        gen_embeddings = [
            self.resnet(gen_subject.unsqueeze(0).to(self.device))
            for gen_subject in gen_subjects
        ]

        similarities = [
            self.cosine_similarity(gen_embedding, ref_embedding)
            for gen_embedding in gen_embeddings
        ]

        return max(similarities)

    def calc_face_separation(self, gen_image):
        face_errors = []
        gen_boxes, gen_probs = self.mtcnn.detect(gen_image, landmarks=False)

        if self.ref_boxes is None or gen_boxes is None:
            return torch.tensor(0.0)

        ref_faces = []
        gen_faces = []
        for j, box in enumerate(self.ref_boxes):
            tensor_face = extract_face(
                self.ref_image,
                box,
            )
            ref_faces.append(tensor_face)
        for j, box in enumerate(gen_boxes):
            tensor_face = extract_face(
                gen_image,
                box,
            )
            gen_faces.append(tensor_face)

        for ref_face, gen_face in zip(ref_faces, gen_faces):
            # デバイスを統一してテンソルに変換
            ref_face = ref_face.to(gen_face.device).float()
            gen_face = gen_face.float()

            # ピクセルごとの MSE を計算
            mse_loss = F.mse_loss(gen_face, ref_face, reduction="mean")
            face_errors.append(mse_loss)

        return torch.stack(face_errors).mean()

    def calc_expression_separation(self, gen_image):
        focus_errors = []
        gen_boxes, gen_probs, gen_points = self.mtcnn.detect(gen_image, landmarks=True)
        if self.ref_boxes is None or gen_boxes is None:
            return torch.tensor(0.0)

        ref_faces = []
        gen_faces = []
        for j, box in enumerate(self.ref_boxes):
            tensor_face = extract_face(
                self.ref_image,
                box,
            )
            ref_faces.append(tensor_face)
        for j, box in enumerate(gen_boxes):
            tensor_face = extract_face(
                gen_image,
                box,
            )
            gen_faces.append(tensor_face)

        for ref_face, gen_face, ref_pts, gen_pts in zip(
            ref_faces, gen_faces, self.ref_points, gen_points
        ):
            ref_face = ref_face.to(gen_face.device).float()
            gen_face = gen_face.float()

            if self.ref_boxes is None or gen_boxes is None:
                focus_errors.append(torch.tensor(0.0))
                continue

            for idx, (ref_point, gen_point) in enumerate(zip(ref_pts, gen_pts)):
                # 各ポイントの周囲の矩形領域を切り取る
                face_rate = 0.1
                ref_focus_region = T.functional.to_tensor(
                    self._extract_focus_region(self.ref_image, ref_point, face_rate)
                )
                gen_focus_region = T.functional.to_tensor(
                    self._extract_focus_region(gen_image, gen_point, face_rate)
                )

                # サイズを合わせる処理
                min_height = min(ref_focus_region.shape[1], gen_focus_region.shape[1])
                min_width = min(ref_focus_region.shape[2], gen_focus_region.shape[2])

                # 小さい方に合わせてクロップ
                ref_focus_region = ref_focus_region[:, :min_height, :min_width]
                gen_focus_region = gen_focus_region[:, :min_height, :min_width]

                # テンソル化してデバイスに移動
                ref_focus_region = ref_focus_region.to(gen_face.device).float()
                gen_focus_region = gen_focus_region.float()

                # ピクセルごとの MSE を計算
                focus_mse_loss = F.mse_loss(
                    gen_focus_region, ref_focus_region, reduction="mean"
                )
                focus_errors.append(focus_mse_loss)

        return torch.stack(focus_errors).mean()

    def _extract_focus_region(self, image, point, face_rate):
        """
        画像からポイントの周囲を face_rate に基づいて矩形領域として切り取る関数。
        """
        x, y = int(point[0]), int(point[1])
        box_size = int(min(image.width, image.height) * face_rate)

        left = min(max(x - box_size // 2, 0), image.width)
        top = min(max(y - box_size // 2, 0), image.height)
        right = max(min(x + box_size // 2, image.width), 0)
        bottom = max(min(y + box_size // 2, image.height), 0)

        try:
            focus_region = image.crop((left, top, right, bottom))
        except Exception as e:
            ic(image.width, image.height)
            ic(x, y, box_size)
            ic(left, top, right, bottom)
            raise e

        return focus_region


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def generate_response_with_images(cfg: DictConfig) -> None:
    if cfg.generated_image_dir is not None:
        generated_image_paths = sorted(
            os.path.join(cfg.generated_image_dir, f)
            for f in os.listdir(cfg.generated_image_dir)
            if (f.endswith(".png") or f.endswith(".jpg")) and not f.startswith("grid")
        )
    else:
        generated_image_paths = [cfg.generated_image]

    print(f"evaluating {len(generated_image_paths)} images...")
    output_dict = [{} for _ in range(len(generated_image_paths))]
    caption_list = cfg.caption_list

    # ref dir の中にディレクトリがあれば
    if cfg.reference_dir is not None:
        reference_image_dirs = [
            os.path.join(cfg.reference_dir, f)
            for f in sorted(os.listdir(cfg.reference_dir))
        ]
        reference_images = [
            os.path.join(reference_image_dir, os.listdir(reference_image_dir)[0])
            for reference_image_dir in reference_image_dirs
        ]
    else:
        reference_images = [cfg.reference_image]

    ic(len(reference_image_dirs))
    ic(len(generated_image_paths))
    ic(len(caption_list))
    assert len(reference_image_dirs) == len(generated_image_paths) // len(caption_list)

    for iter_idx, reference_image in enumerate(reference_images):
        evaluator = Evaluator(reference_image)

        start = iter_idx * len(caption_list)
        end = (iter_idx + 1) * len(caption_list)

        for idx, (generated_image_path, caption) in enumerate(
            zip(generated_image_paths[start:end], caption_list)
        ):
            caption = caption.replace(" <|image|>", "")
            # print(f"Processing {generated_image_path}: {caption}...")
            ic(idx, reference_image, generated_image_path, caption)

            gen_image = Image.open(generated_image_path)

            sim_txt2img = evaluator.calc_sim_txt2img(caption, gen_image)
            sim_img2img = evaluator.calc_sim_img2img(gen_image)
            # face_separation = evaluator.calc_face_separation(gen_image)
            # expression_separation = evaluator.calc_expression_separation(gen_image)

            # print(f"txt2img >> {sim_txt2img}")
            # print(f"img2img >> {sim_img2img}")
            # print(f"face separation >> {face_separation}")
            # print(f"expression separation >> {expression_separation}")

            output_dict[start + idx]["image_path"] = generated_image_path
            output_dict[start + idx]["caption"] = caption
            output_dict[start + idx]["sim_txt2img"] = sim_txt2img
            output_dict[start + idx]["sim_img2img"] = sim_img2img
            # output_dict[start + idx]["face_separation"] = face_separation.item()
            # output_dict[start + idx]["expression_separation"] = (
            #     expression_separation.item()
            # )

    df = pd.DataFrame(output_dict)
    df.to_csv(cfg.save_csv_path, index=False)

    print(
        f"mean score txt2img >> {df['sim_txt2img'].mean():.3f}({df['sim_txt2img'].std():.3f})"
    )
    print(
        f"mean score img2img >> {df['sim_img2img'].mean():.3f}({df['sim_img2img'].std():.3f})"
    )
    # print(
    #     f"mean score face separation >> {df['face_separation'].mean():.3f}({df['face_separation'].std():.3f})"
    # )
    # print(
    #     f"mean score expression separation >> {df['expression_separation'].mean():.3f}({df['expression_separation'].std():.3f})"
    # )


if __name__ == "__main__":
    generate_response_with_images()
