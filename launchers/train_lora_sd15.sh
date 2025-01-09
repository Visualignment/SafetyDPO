# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~30 hours / 200 steps

accelerate launch --mixed_precision="fp16"  train.py \
  --pretrained_model_name_or_path="sd-legacy/stable-diffusion-v1-5" \
  --dataset_name="datasets/nemo_captions-pickapic_formatted" \
  --train_batch_size=4 \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-5 --scale_lr \
  --cache_dir="./export/share/datasets/vision_language/pick_a_pic_v2/" \
  --checkpointing_steps 500 \
  --beta_dpo 5000 \
  --output_dir="trained_models/nemo_captions-pickapic_formatted"
