#! /bin/bash
#SBATCH --job-name=ToF_DAR_test
#SBATCH -o ./scripts/out/ToF_DAR_test.log
#SBATCH --partition=ampereq
#SBATCH -J ToF_DAR_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

nvidia-smi
now=$(date +"%Y-%m-%d %H:%M:%S")

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

source activate deltar

# dataset/config
experiment_name=ToF_DAR_test
RUN_ROOT=${RUN_ROOT:-$HOME/tof-dar-runs}
path="${RUN_ROOT}/exp/${experiment_name}"
LOG_DIR="${RUN_ROOT}/scripts/out"
mkdir -p "$path" "$LOG_DIR"

# data
DATA_ROOT=./data/ZJUL5
JSON_FILE=./data/ZJUL5/data.json
SPLIT=test
IMAGE_SIZE=256
DEPTH_MIN=0.0
DEPTH_MAX=10.0

# run a short sanity pass
BATCH_SIZE=1
STEPS=10
LR=0.0002
DEVICE=cuda
VAR_CKPT=./var_d16.pth

python3 train_tof_depth_var.py \
  --data_root "$DATA_ROOT" \
  --json_file "$JSON_FILE" \
  --split "$SPLIT" \
  --image_size $IMAGE_SIZE \
  --depth_min $DEPTH_MIN \
  --depth_max $DEPTH_MAX \
  --batch_size $BATCH_SIZE \
  --steps $STEPS \
  --lr $LR \
  --device $DEVICE \
  --var_ckpt "$VAR_CKPT" \
  2>&1 | tee -a "${LOG_DIR}/${now}.log"
