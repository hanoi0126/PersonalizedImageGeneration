import os

import hydra
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from omegaconf import DictConfig
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class Evaluator:
    def __init__(self, ref_image_path):
        # デバイス設定（GPUが利用可能な場合にはGPUに設定）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.mtcnn = MTCNN(margin=0, keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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

    def get_subject_from_image(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.mtcnn(image)

    def calc_sim_img2img(self, gen_image):
        gen_subjects = self.get_subject_from_image(gen_image)
        if gen_subjects is None:
            print("No face detected in the generated image")
            return [0], 0

        subject_num = len(gen_subjects)

        print(f"subject num >> {subject_num}")

        ref_embedding = self.resnet(self.ref_face[0].to(self.device).unsqueeze(0))
        gen_embeddings = [self.resnet(gen_subject.unsqueeze(0).to(self.device)) for gen_subject in gen_subjects]

        similarities = [
            self.cosine_similarity(gen_embedding, ref_embedding)
            for gen_embedding in gen_embeddings
        ]

        return max(similarities)


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

    evaluator = Evaluator(cfg.reference_image)

    for idx, (generated_image_path, caption) in enumerate(
        zip(generated_image_paths, caption_list)
    ):
        caption = caption.replace("<|image|>", "")
        print(f"Processing {generated_image_path}: {caption}...")

        gen_image = Image.open(generated_image_path)

        sim_txt2img = evaluator.calc_sim_txt2img(caption, gen_image)
        sim_img2img = evaluator.calc_sim_img2img(gen_image)

        print(f"txt2img >> {sim_txt2img}")
        print(f"img2img >> {sim_img2img}")

        output_dict[idx]["image_path"] = generated_image_path
        output_dict[idx]["caption"] = caption
        output_dict[idx]["sim_txt2img"] = sim_txt2img
        output_dict[idx]["sim_img2img"] = sim_img2img

    df = pd.DataFrame(output_dict)
    df.to_csv(cfg.save_csv_path, index=False)

    print(f"mean score txt2img >> {df['sim_txt2img'].mean()}({df['sim_txt2img'].std()})")
    print(f"mean score img2img >> {df['sim_img2img'].mean()}({df['sim_img2img'].std()})")


if __name__ == "__main__":
    generate_response_with_images()
