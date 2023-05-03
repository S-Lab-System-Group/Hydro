GPU_ID=$1
SCALE=$2
MODEL='resnet18'
DATASET='cifar10'
BS=256

TEST_LIST=(1 10 20 25)
if [[ ${SCALE} -eq 8 ]]; then
  TEST_LIST+=(100 150 220)
fi

echo "${TEST_LIST}"

for FUSION in "${TEST_LIST[@]}"
do 
  echo -e "\033[1mclean python processes\033[0m"
  sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
  python 1_fusion_efficiency.py \
  --model ${MODEL} \
  --dataset ${DATASET} \
  --batch-size ${BS} \
  --fusion-num ${FUSION} \
  --scale ${SCALE} \
  --gpu ${GPU_ID} \
  --save_file "${MODEL}_${BS}.csv" 
done

