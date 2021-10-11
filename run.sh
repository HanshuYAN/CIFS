#!/bin/bash

# Resnet 18
CUDA_VISIBLE_DEVICES=0 python CIFAR10_Res18_CIFS.py --is_Train --GPU_IDs 0 --network _  --exp_path ./experiments/Cln_Res18_cifar10 --attack_loss CE
CUDA_VISIBLE_DEVICES=1 python CIFAR10_Res18_CIFS.py --is_Train --is_AdvTr --GPU_IDs 0 --network _  --exp_path ./experiments/PAT_Res18_cifar10 --attack_loss CE
CUDA_VISIBLE_DEVICES=0 python CIFAR10_Res18_CIFS_test.py --GPU_IDs 0 --network _  --which_model ./experiments/...

# CAS (Layer4)
CUDA_VISIBLE_DEVICES=0 python CIFAR10_Res18_CIFS.py --is_Train --is_AdvTr --GPU_IDs 0 --network CAS_L4 --exp_path ./experiments/...
CUDA_VISIBLE_DEVICES=0 python CIFAR10_Res18_CIFS_test.py --GPU_IDs 0 --network CAS_L4 --which_model ./experiments/...pt --is_attack 1 --is_joint 1 --beta_atk 2

#  CIFS
CUDA_VISIBLE_DEVICES=1 python CIFAR10_Res18_CIFS.py --is_Train --GPU_IDs 0 --network CIFS_L4 --exp_path ./experiments/Cln_Res18_cifar10_CIFS
CUDA_VISIBLE_DEVICES=0 python CIFAR10_Res18_CIFS.py --is_Train --is_AdvTr --GPU_IDs 0 --network CIFS_L4 --exp_path ./experiments/PAT_Res18_cifar10_CIFS --attack_loss Joint --beta_atk 2 \
                                                                                        --resume --checkpoint ./experiments/Cln_Res18_cifar10_CIFS/nets/ckp_best.pt

# CUDA_VISIBLE_DEVICES=0 python CIFAR10_Res18_CIFS_test.py --GPU_IDs 0 --network CSAFR_topk --which_model ./experiments/....pt --is_attack 1 --is_joint 1 --beta_atk 2















