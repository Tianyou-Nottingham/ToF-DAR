#! /bin/bash
#SBATCH --job-name=ToF_DAR_train_largeBS
#SBATCH -o ./scripts/out/ToF_DAR_train_largeBS.log
#SBATCH --partition=ampereq
#SBATCH -J ToF_DAR_train_largeBS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

nvidia-smi
now=$(date +"%Y-%m-%d %H:%M:%S")
source activate deltar

PORT=29504
gpus=1

# dataset/config
experiment_name=ToF_DAR_train_largeBS
path=exp/ToF_DAR_train_largeBS
mkdir -p "$path" 

# data
DATASET=nyu
DATA_ROOT=./data/nyu_sync
JSON_FILE=./train_test_split/nyu_sync.json
DATASET_EVAL=nyu
DATA_ROOT_EVAL=./data/nyu_depth_v2/official_splits/test
JSON_FILE_EVAL=./train_test_split/nyu_sync.json
SPLIT=train
IMAGE_SIZE=256
DEPTH_MIN=0.0
DEPTH_MAX=10.0
INPUT_HEIGHT=416
INPUT_WIDTH=544
TRAIN_ZONE_NUM=8
ZONE_SAMPLE_NUM=16
SIMU_MAX_DISTANCE=4.0
DROP_HIST=0.0
NOISE_MEAN=0.0
NOISE_SIGMA=0.0
NOISE_PROB=0.0

# train
BATCH_SIZE=16
MODEL_DEPTH=16
EPOCHS=20
EVAL_EVERY=1
EVAL_SPLIT=test
LR=0.00001
DEVICE=cuda
VAR_CKPT=./var_d16.pth
VAE_CKPT=vae_ch160v4096z32.pth
VIS_EVERY=0

# optional ZJUL5 validation
ZJU_DATA_ROOT=./data/ZJUL5
ZJU_JSON_FILE=./data/ZJUL5/data.json
ZJU_EVAL_SPLIT=test

python3  -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$PORT \
    train_tof_depth_var.py \
    --dataset "$DATASET" \
    --data_root "$DATA_ROOT" \
    --json_file "$JSON_FILE" \
    --dataset_eval "$DATASET_EVAL" \
    --data_root_eval "$DATA_ROOT_EVAL" \
    --json_file_eval "$JSON_FILE_EVAL" \
    --epochs $EPOCHS \
    --eval_every $EVAL_EVERY \
    --eval_split $EVAL_SPLIT \
    --split "$SPLIT" \
    --image_size $IMAGE_SIZE \
    --depth_min $DEPTH_MIN \
    --depth_max $DEPTH_MAX \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --device $DEVICE \
    --input_height $INPUT_HEIGHT \
    --input_width $INPUT_WIDTH \
    --train_zone_num $TRAIN_ZONE_NUM \
    --zone_sample_num $ZONE_SAMPLE_NUM \
    --simu_max_distance $SIMU_MAX_DISTANCE \
    --drop_hist $DROP_HIST \
    --noise_mean $NOISE_MEAN \
    --noise_sigma $NOISE_SIGMA \
    --noise_prob $NOISE_PROB \
    --vae_ckpt "$VAE_CKPT" \
    --vis_every $VIS_EVERY \
    --vis_dir "$VIS_DIR" \
    --zju_data_root "$ZJU_DATA_ROOT" \
    --zju_json_file "$ZJU_JSON_FILE" \
    --zju_eval_split "$ZJU_EVAL_SPLIT" \
    2>&1 | tee -a "${LOG_DIR}/${now}.log"
