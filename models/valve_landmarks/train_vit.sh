#! /bin/bash
#SBATCH --nodes=1
#SBATCH -J train_vit
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH -p small


eval "$(conda shell.bash hook)"
conda activate monai

set -v

export CUDA_VISIBLE_DEVICES=0
export BUNDLE="$(pwd)/"
export PYTHONPATH="$BUNDLE"

CKPT=none
CKPT="./results/output_230630_210555/net_key_metric=-47.9900.pt"

DATAFILE="/data/LargeArchiveStorage/valvelandmarks.npz"

PYTHON=python

cat "$BUNDLE/configs/common.yaml"
cat "$BUNDLE/configs/train_vit.yaml"

python -m monai.bundle run training \
    --meta_file configs/metadata.json \
    --config_file configs/train_vit.yaml \
    --bundle_root $BUNDLE \
    --ckpt_path $CKPT \
    --dataset_file $DATAFILE
    