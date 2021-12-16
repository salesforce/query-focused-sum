SPLIT=test
export ROUGE_HOME=/export/home/query-focused-conv-summ/rouge/ROUGE-1.5.5/
for NAME in \
  qmsum_32_512_strided_wikisum \
  qmsum_32_512_strided
do
  echo "************************************************************"
  echo $NAME
  python ../rouge/report_rouge.py \
    --ref-path ../data/${SPLIT}.target \
    --pred-paths \
        output/${NAME}_1/selected_checkpoint/predictions.${SPLIT} \
        output/${NAME}_2/selected_checkpoint/predictions.${SPLIT} \
        output/${NAME}_3/selected_checkpoint/predictions.${SPLIT} \
        output/${NAME}_4/selected_checkpoint/predictions.${SPLIT} \
        output/${NAME}_5/selected_checkpoint/predictions.${SPLIT}
done