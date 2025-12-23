# NCCL settings optimized for DGX with NVLink
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL

# Update these paths to point to your WebDataset tar files
# Expected format: tar archives with paired .json + image files
TRAIN_DATA="/path/to/your/train-*.tar"
EVAL_DATA="/path/to/your/eval-*.tar"

accelerate launch \
  --config_file acc.yaml \
  sft_clean.py \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --max-length 3072 \
  --train-vision \
  --tar-pattern "${TRAIN_DATA}" \
  --eval-tar-pattern "${EVAL_DATA}" \
  --output-dir qwen25vl-3B-OCR-SFT \
  --batch-size 8 \
  --grad-accum 2 \
  --epochs 1 \
  --learning-rate 1e-6 \
  --bf16 \
  --num-workers 0 \
  --persistent-workers False \
  --prefetch-factor 0 \
  --deepspeed-config ds_zero3.json \
  --val-samples-per-epoch 2048
