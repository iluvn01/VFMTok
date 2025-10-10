# !/bin/bash
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=8
model_type='GPT-3B'
rm -rf engine/__pycache__ tokenizer/tokenizer_image/__pycache__
scripts/autoregressive/torchrun.sh test_net.py --vq-ckpt pretrained_models/vit_vqgan_step_225000.pt            \
    --gpt-ckpt snapshot/model_dump/${model_type}-$1.pt --compile --gpt-model ${model_type} --image-size 336 \
    --sample-dir samples --image-size-eval 256 --cfg-scale $2 --precision bf16 --per-proc-batch-size $3   \
    --codebook-slots-embed-dim 12 --latent-size 16 2>&1 | tee 'hello.log'
