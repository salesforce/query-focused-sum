
$CHUNKS=$1
$OUTPUT_DIR=$2


# Convert data to format required for RelReg training and inference; 0 for utterance-level data
python prep_data_relreg.py $CHUNKS

# Train RelReg on utterance-level input
CUDA_VISIBLE_DEVICES=0 python transformers/examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path google/electra-large-discriminator \
  --train_file ../data/train.relreg.csv \
  --validation_file ../data/val.relreg.csv \
  --save_steps 3000 \
  --do_train \
  --do_eval \
  --max_seq_length 384 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --save_total_limit 1 \
  --output_dir ./${OUTPUT_DIR} ; 

# Run inference inference
for split in 'train' 'val' 'test'
do
    CUDA_VISIBLE_DEVICES=0 python transformers/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path ./${OUTPUT_DIR} \
    --train_file ../data/train.relreg.csv \
    --validation_file ../data/val.relreg.csv \
    --test_file ../data/${split}.relreg.csv \
    --save_steps 3000 \
    --do_predict \
    --max_seq_length 384 \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir ./${OUTPUT_DIR}-${split} ; 
done

# Collect predictions and process to format for seq2seq models; 0 signifies not using the semgneted input
python postprocess_relreg.py $CHUNKS $OUTPUT_DIR