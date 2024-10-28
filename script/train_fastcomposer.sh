if [ -f .env ]; then
    export $(cat .env | xargs)
fi

git config --global --add safe.directory /workspace

poetry run accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11135 \
    --num_processes 2 \
    --multi_gpu \
    fastcomposer/train_fc.py d
