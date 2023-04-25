GPU_ID=$1
CO=$2
MODEL='resnet18'
DATASET='cifar10'
BS=256
CO_REST=$(( ${CO} - 1 ))

echo -e "\033[1mclean python processes\033[0m"
echo -e "\033[1m${CO} mps\033[0m"
python 1_fusion_efficiency.py \
--model ${MODEL} \
--dataset ${DATASET} \
--batch-size ${BS} \
--fusion-num 1 \
--scale 1 \
--epochs 4 \
--save_file "${MODEL}_mps.csv" \
--gpu ${GPU_ID} \
--co ${CO} &
for ((i=1;i<=${CO_REST};i++));
do 
if [ ${i} == ${CO_REST} ]
then
python 1_fusion_efficiency.py \
--model ${MODEL} \
--dataset ${DATASET} \
--batch-size ${BS} \
--fusion-num 1 \
--scale 1 \
--epochs 4 \
--gpu ${GPU_ID} 
else
python 1_fusion_efficiency.py \
--model ${MODEL} \
--dataset ${DATASET} \
--batch-size ${BS} \
--fusion-num 1 \
--scale 1 \
--epochs 4 \
--gpu ${GPU_ID} &
fi
done
