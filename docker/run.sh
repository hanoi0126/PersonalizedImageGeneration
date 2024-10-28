#!/bin/bash

IMAGE_NAME="multi-composer"
CONTAINER_NAME="multi-composer"
GPU_NUMBER=""
USE_ALL_GPUS=false
CMD="/bin/bash"

# Parse the arguments
while getopts ":g:a" opt; do
    case ${opt} in
        g )
            GPU_NUMBER="${OPTARG}"
            ;;
        a )
            USE_ALL_GPUS=true
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        : )
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

# Check if the GPU number or "all" option is provided
if [ -z "$GPU_NUMBER" ] && [ "$USE_ALL_GPUS" = false ]; then
    echo "Please provide the GPU number using the -g option or use -a to use all GPUs."
    exit 1
fi

# Set the container name
if [ "$USE_ALL_GPUS" = true ]; then
    CONTAINER_NAME="${CONTAINER_NAME}-all"
else
    CONTAINER_NAME="${CONTAINER_NAME}-${GPU_NUMBER}"
fi

echo "$(pwd)/MultiComposer:/workspace"

# Run the docker command
if [ "$USE_ALL_GPUS" = true ]; then
    docker run -it --gpus all \
        -v $(pwd)/MultiComposer:/workspace \
        --ipc=host \
        --name $CONTAINER_NAME \
        $IMAGE_NAME \
        $CMD
else
    docker run -it --gpus device=$GPU_NUMBER \
        -v $(pwd)/MultiComposer:/workspace \
        --ipc=host \
        --name $CONTAINER_NAME \
        $IMAGE_NAME \
        $CMD
fi
