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

CKPT=""

DATAFILE="/data/LargeArchiveStorage/valvelandmarks.npz"

PYTHON=python

cat "$BUNDLE/configs/common.yaml"
cat "$BUNDLE/configs/train.yaml"

python -m monai.bundle run training \
    --meta_file configs/metadata.json \
    --config_file configs/train.yaml \
    --bundle_root $BUNDLE \
    --num_epochs 20 \
    '--network_def#channels' '[8, 16, 32, 64, 128]' \
    --ckpt_path none \
    '--handlers#3#_disabled_' true \
    --dataset_file $DATAFILE
    