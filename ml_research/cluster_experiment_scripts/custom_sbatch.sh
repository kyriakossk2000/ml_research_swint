#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:


source /lustre/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..
python train_evaluate_emnist_classification_system.py --batch_size 32 --continue_from_epoch -1 --seed 0 --image_num_channels 3 --image_height 224 --image_width 32 --dim_reduction_type "strided" --num_layers 4 --num_filters 64 --num_epochs 10 --experiment_name 'product10k_vit_b_16_exp_eval' --use_gpu "True" --weight_decay_coefficient 0 --dataset_name "Product10k" --model_to_load "vit_b_16" --eval_mode_on "True" --eval_model_checkpoint "vit_b_16" --eval_model_name "vit_b_16"