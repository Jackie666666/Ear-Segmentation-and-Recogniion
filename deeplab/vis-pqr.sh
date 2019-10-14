cd ..

# Set up the working environment.
CURRENT_DIR=$(pwd)
PQR_FOLDER="PQR"
DATASET_DIR="datasets"
EXP_FOLDER="exp/train_on_trainval_set"

WORK_DIR="${CURRENT_DIR}/deeplab"
CHECKPOINT="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/train"
VIS_DIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/vis"
DATASET="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/tfrecord"



python deeplab/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size=960 \
    --vis_crop_size=1280 \
    --dataset="pqr" \
    --checkpoint_dir="${CHECKPOINT}\model.ckpt-500" \
    --vis_logdir=${VIS_DIR} \
    --dataset_dir=${DATASET} 