$CHUNKS=$1
$OUTPUT_DIR=$2

# Run relreg-tt with (0/1) utterance/segments and max encoder length of 256
python add_rouge.py $CHUNKS
python train_relreg_tt.py nli-distilroberta-base-v2 $CHUNKS $OUTPUT_DIR 256     
python test_relreg_tt.py $OUTPUT_DIR $CHUNKS $OUTPUT_DIR 256     
