CUDA_VISIBLE_DEVICES=0 python 2_scaling_fidelity.py --scale 1 & 
CUDA_VISIBLE_DEVICES=1 python 2_scaling_fidelity.py --scale 2 &
CUDA_VISIBLE_DEVICES=2 python 2_scaling_fidelity.py --scale 4 &
CUDA_VISIBLE_DEVICES=3 python 2_scaling_fidelity.py --scale 8

sleep 3
ray stop --force