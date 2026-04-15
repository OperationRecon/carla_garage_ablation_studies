#!/bin/bash

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
export CARLA_ROOT=/home/mohamednaas/Documents/carla_garage/carla

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/software/anaconda3/lib

export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=1

# -----------------------------------------------------------------------------
# Experiment config
# -----------------------------------------------------------------------------
ID="tfpp_0010_0"
LOGDIR="/media/mohamednaas/hardDisk/augmented_model"
ROOT_DIR="/home/mohamednaas/Documents/carla_project/dataset_small_model/data"

INITIAL_LOAD_FILE='media/mohamednaas/hardDisk/augmented_model/tfpp_0010_0/model_0030.pth'

TARGET_EPOCHS=31
CHUNK=2
MAX_RETRIES=5
START_EPOCH=0
FIRST_CHUNK=1

# -----------------------------------------------------------------------------
# Training arguments (base)
# -----------------------------------------------------------------------------
BASE_ARGS=(
    --id ${ID}
    --lr 0.0002
    --batch_size 7
    --root_dir ${ROOT_DIR}
    --logdir ${LOGDIR}
    --setting all

    # Model / training switches
    --backbone transFuser
    --image_architecture regnety_008
    --lidar_architecture regnety_008
    --n_layer 1
    --use_velocity 1
    --use_controller_input_prediction 1
    --use_wp_gru 0

    # Schedule / optimization
    --schedule_reduce_epoch_01 30
    --schedule_reduce_epoch_02 40
    --weight_decay 0.01
    --use_cosine_schedule 1

    # Data / loader
    --val_every 5
    --sync_batch_norm 0
    --zero_redundancy_optimizer 1
    --use_disk_cache 0
    --lidar_seq_len 1
    --realign_lidar 1
    --use_ground_plane 0
    --pred_len 8
    --train_sampling_rate 1
    --num_repetitions 1

    # Augmentation / training behavior
    --augment 1
    --use_color_aug 1
    --use_cutout 0

    # Loss / tasks
    --use_semantic 1
    --use_depth 1
    --detect_boxes 1
    --use_bev_semantic 1
    --use_focal_loss 0
    --use_label_smoothing 0
    --estimate_class_distributions 0
    --estimate_semantic_distribution 0

    # Architecture extras
    --gru_hidden_size 32
    --gru_input_size 256
    --bev_down_sample_factor 4
    --perspective_downsample_factor 1
    --transformer_decoder_join 1
    --add_features 1
    --freeze_backbone 0
    --learn_multi_task_weights 0

    # Sensors / BEV
    --max_height_lidar 100.0
    --max_num_bbs 30
    --smooth_route 1
    --use_tp 1
    --tp_attention 0
    --multi_wp_output 0

    # System
    --local_rank -999
    --cpu_cores 12
    --seed 0
    --use_amp 0
    --use_grad_clip 0

    # Misc
    --use_plant 0
    --learn_origin 1
    --input_path_to_target_speed_network 0
    --predict_checkpoint_len 10
    --crop_image 1
    --max_x 32
    --crop_bev_height_only_from_behind 0
    --lidar_resolution_height 256
    --dataset_cache_name 20
    --cosine_t0 1
    --compile 0
    --compile_mode default
    --chnlswap 1
)

# -----------------------------------------------------------------------------
# Helper: parse epoch from checkpoint filename
# -----------------------------------------------------------------------------
parse_epoch_from_file() {
    f=$(basename "$1")
    num=$(echo "$f" | grep -o -E '[0-9]+' | tail -n1 || true)
    if [ -n "$num" ]; then
        echo $((10#$num))
    else
        echo -1
    fi
}

# -----------------------------------------------------------------------------
# Load checkpoint logic
# -----------------------------------------------------------------------------
if [ -n "${INITIAL_LOAD_FILE}" ] && [ -f "${INITIAL_LOAD_FILE}" ]; then
    loaded_epoch=$(parse_epoch_from_file "${INITIAL_LOAD_FILE}")

    if [ "${loaded_epoch}" -eq $((TARGET_EPOCHS - 1)) ]; then
        echo "Checkpoint is final epoch (${loaded_epoch}). Starting fresh."
        loaded_epoch=-1
        load_file_to_use=""
    else
        load_file_to_use="${INITIAL_LOAD_FILE}"
    fi

elif [ "${START_EPOCH}" -gt 0 ]; then
    loaded_epoch=$((START_EPOCH - 1))
    candidate="${LOGDIR}/${ID}/model_$(printf "%04d" "${loaded_epoch}").pth"

    if [ -f "${candidate}" ]; then
        load_file_to_use="${candidate}"
    else
        echo "Missing checkpoint ${candidate}, starting fresh."
        load_file_to_use=""
    fi
else
    loaded_epoch=-1
    load_file_to_use=""
fi

start_epoch=$((loaded_epoch + 1))
echo "Starting from epoch ${start_epoch}, target ${TARGET_EPOCHS}"

# -----------------------------------------------------------------------------
# Training loop (chunked)
# -----------------------------------------------------------------------------
while [ ${start_epoch} -lt ${TARGET_EPOCHS} ]; do

    chunk_end=$((start_epoch + CHUNK))
    if [ ${chunk_end} -gt ${TARGET_EPOCHS} ]; then
        chunk_end=${TARGET_EPOCHS}
    fi

    if [ -n "${load_file_to_use}" ]; then
        load_arg=(--load_file "${load_file_to_use}")
    else
        load_arg=()
    fi

    echo "Chunk: ${start_epoch} → $((chunk_end - 1))"

    ARGS=("${BASE_ARGS[@]}")

    # force continue_epoch after first chunk if needed
    if [ "${FIRST_CHUNK}" -eq 0 ]; then
        for i in "${!ARGS[@]}"; do
            if [ "${ARGS[$i]}" = "--continue_epoch" ] && [ "${ARGS[$((i+1))]}" = "0" ]; then
                ARGS[$((i+1))]=1
            fi
        done
    fi

    retries=0
    while true; do
        /home/mohamednaas/miniconda3/envs/garage_2/bin/python \
        -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=1 \
        --max_restarts=1 \
        --rdzv_id=42353467 \
        --rdzv_backend=c10d \
        /home/mohamednaas/Documents/carla_garage/team_code/train.py \
        "${ARGS[@]}" \
        "${load_arg[@]}" \
        --epochs ${chunk_end}

        ret=$?
        if [ ${ret} -eq 0 ]; then
            echo "Chunk done."
            FIRST_CHUNK=0
            break
        fi

        retries=$((retries+1))
        if [ ${retries} -ge ${MAX_RETRIES} ]; then
            echo "Chunk failed too many times." >&2
            exit 1
        fi

        echo "Retry ${retries}/${MAX_RETRIES}"
        sleep 10
    done

    expected_model="${LOGDIR}/${ID}/model_$(printf "%04d" $((chunk_end - 1))).pth"

    wait_time=0
    while [ ! -f "${expected_model}" ] && [ ${wait_time} -lt 300 ]; do
        echo "Waiting checkpoint... (${wait_time}s)"
        sleep 5
        wait_time=$((wait_time+5))
    done

    if [ -f "${expected_model}" ]; then
        load_file_to_use="${expected_model}"
        start_epoch=$((chunk_end))
    else
        echo "Missing checkpoint, aborting." >&2
        exit 1
    fi

done

echo "Training complete."