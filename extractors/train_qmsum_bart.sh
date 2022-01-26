NAME=relreg-qmsum-256-wikisum
NUM_RUNS=5
START=1
for RUN in $(seq $START $NUM_RUNS)
do
 CUDA_VISIBLE_DEVICES=0 python -u ../multiencoder/train.py \
 --train_file $RELREG_OUTPUT_DIR/train.csv \
 --validation_file $RELREG_OUTPUT_DIR/val.csv \
 --do_train \
 --do_eval \
 --learning_rate 0.000005 \
 --model_name_or_path $PATH_TO_CHECKPOINT \
 --metric_for_best_model eval_mean_rouge \
 --output_dir output/${NAME}_${RUN} \
 --per_device_train_batch_size 4 \
 --max_source_length 1024 \
 --generation_max_len 256 \
 --val_max_target_length 256 \
 --overwrite_output_dir \
 --per_device_eval_batch_size 4 \
 --predict_with_generate \
 --evaluation_strategy epoch \
 --num_train_epochs 10 \
 --save_strategy epoch \
 --logging_strategy epoch \
 --load_best_model_at_end \
 --compute_rouge_for_train True \
 --seed $RUN &> ${NAME}_${RUN}.out
done