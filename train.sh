EVAL_STEP=200

CUDA_VISIBLE_DEVICES=1 python run_swag.py \
    --model_name_or_path roberta-large \
    --output_dir /home/khangln/JAIST_DRIVE/WORK/IconQA/my/models/roberta-large-train-both-img-txt \
    --do_train \
    --do_eval \
    --train_file /home/khangln/JAIST_DRIVE/WORK/IconQA/my/data/combine_img_txt/train.json \
    --validation_file /home/khangln/JAIST_DRIVE/WORK/IconQA/my/data/combine_img_txt/val.json \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 10 \
    --per_gpu_eval_batch_size=16 \
    --per_device_train_batch_size=16 \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEP \
    --save_steps $EVAL_STEP \
    --logging_strategy steps \
    --logging_steps $EVAL_STEP \
    --load_best_model_at_end \
    --report_to tensorboard
