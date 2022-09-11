EVAL_STEP=200

CUDA_VISIBLE_DEVICES=1 python run_swag.py \
    --model_name_or_path /home/khangln/JAIST_DRIVE/WORK/IconQA/my/models/roberta-large-tune-test/checkpoint-12800 \
    --output_dir /home/khangln/JAIST_DRIVE/WORK/IconQA/my/models/predict \
    --do_eval \
    --train_file /home/khangln/JAIST_DRIVE/WORK/IconQA/my/data/choose_txt/train_captioned_iconqa_choosetxt.json \
    --validation_file /home/khangln/JAIST_DRIVE/WORK/IconQA/my/test_skill_split/test_counting.json \
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
