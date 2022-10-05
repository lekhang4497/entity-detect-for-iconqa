EVAL_STEP=200

CUDA_VISIBLE_DEVICES=1 python eval_multi_choice.py \
    --model_name_or_path models/roberta-large-train-both-img-txt/checkpoint-44400 \
    --output_dir models/predict \
    --do_eval \
    --train_file data/choose_img/test_captioned_iconqa_choose_img_with_ids.json \
    --validation_file data/choose_img/test_captioned_iconqa_choose_img_with_ids.json \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 6 \
    --per_gpu_eval_batch_size=16 \
    --per_device_train_batch_size=16 \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEP \
    --save_steps $EVAL_STEP \
    --logging_strategy steps \
    --logging_steps $EVAL_STEP \
    --load_best_model_at_end \
    --report_to tensorboard
