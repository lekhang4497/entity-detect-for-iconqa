EVAL_STEP=200

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
    --model_name_or_path facebook/bart-large \
    --output_dir seq2seq_models/bart-large-train-fill-blank-2 \
    --do_train \
    --do_eval \
    --train_file data/fill_in_blank/train_captioned_iconqa_fill_in_blank.json \
    --validation_file data/fill_in_blank/val_captioned_iconqa_fill_in_blank.json \
    --text_column source \
    --summary_column target \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --predict_with_generate \
    --metric_for_best_model exact_match \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEP \
    --save_steps $EVAL_STEP \
    --logging_strategy steps \
    --logging_steps $EVAL_STEP \
    --load_best_model_at_end \
    --report_to tensorboard
