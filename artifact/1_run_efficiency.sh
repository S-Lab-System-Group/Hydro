CUDA_VISIBLE_DEVICES=0 bash ./1_fusion.sh 0 8 &
CUDA_VISIBLE_DEVICES=1 bash ./1_fusion.sh 1 1 &
CUDA_VISIBLE_DEVICES=2 bash ./1_mps.sh 2 5 &
CUDA_VISIBLE_DEVICES=3 bash ./1_mps.sh 3 10 &

