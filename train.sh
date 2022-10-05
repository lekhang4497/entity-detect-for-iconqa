EVAL_STEP=200

CUDA_VISIBLE_DEVICES=0 python run_swag.py \
    --model_name_or_path roberta-large \
    --output_dir models/roberta-large-train-img \
    --do_train \
    --do_eval \
    --train_file data/choose_img/train_captioned_iconqa_multichoice.json \
    --validation_file data/choose_img/val_captioned_iconqa_multichoice.json \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 15 \
    --per_gpu_eval_batch_size=16 \
    --per_device_train_batch_size=16 \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEP \
    --save_steps $EVAL_STEP \
    --logging_strategy steps \
    --logging_steps $EVAL_STEP \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --report_to tensorboard
