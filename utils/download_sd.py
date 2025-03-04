import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

os.makedirs("./models/basemodel", exist_ok=True)

model_name = "benjamin-paine/stable-diffusion-v1-5"
download_path = snapshot_download(
    repo_id=model_name,
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
    local_dir="./models/basemodel",
    local_dir_use_symlinks=False,
)
