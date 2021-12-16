for MODEL_NAME in  \
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
  python select_checkpoints.py output/$MODEL_NAME
done