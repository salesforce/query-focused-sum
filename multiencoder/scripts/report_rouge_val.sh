SPLIT=val
export ROUGE_HOME=/export/home/query-focused-conv-summ/rouge/ROUGE-1.5.5/
for NAME in \
  qmsum_16_256_strided \
  qmsum_32_256_strided \
  qmsum_64_256_strided \
  qmsum_8_512_strided \
  qmsum_16_512_strided \
  qmsum_32_512_strided \
  qmsum_4_1024_strided \
  qmsum_8_1024_strided \
  qmsum_16_1024_strided \
  qmsum_32_512_nostrided \
  qmsum_32_512_strided_wikisum
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