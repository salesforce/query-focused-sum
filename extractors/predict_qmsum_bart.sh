NAME=relreg-qmsum-256-wikisum
SPLIT=val
NUM_RUNS=5
START=1
for RUN in $(seq $START $NUM_RUNS)
do
  OUTPUT_DIR=output/${NAME}_${RUN}
  CUDA_VISIBLE_DEVICES=0 python -u train.py   
    --test_file $RELREG_OUTPUT_DIR/val.csv    
    --do_predict     
    --model_name_or_path $OUTPUT_DIR/selected_checkpoint     
    --output_dir ${OUTPUT_DIR}/selected_checkpoint/predition_logs_${SPLIT}     
    --prediction_path ${OUTPUT_DIR}/selected_checkpoint/predictions.${SPLIT}     
    --max_source_length 512     
    --generation_max_len 256     
    --val_max_target_length 256     
    --overwrite_output_dir     
    --per_device_eval_batch_size 4     
    --predict_with_generate
done
