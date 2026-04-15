export CARLA_ROOT=/home/mohamednaas/Documents/carla_garage/carla
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mohamednaas/miniconda3/lib

export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=1

python /home/mohamednaas/Documents/carla_garage/team_code/activation_extractor.py \
    --logdir /media/mohamednaas/hardDisk/activation_extraction_output \
    --load_file /media/mohamednaas/hardDisk/augmented_model/tfpp_0010_0/model_0030.pth \
    --root_dir /media/mohamednaas/hardDisk/dataset \
    --cpu_cores 12 \
    --batch_size 20 \
    --num_batches 20 \
    --overlay_only 1 \
    --save_activations /media/mohamednaas/hardDisk/activation_extraction_output_singular_file_no_attention_gating/channel_swap_with_with_swappin_on_input \
    --sample_indices 0,2,4,6,8,10,12,14,16,18,20,22 \
    --config_json /home/mohamednaas/Documents/carla_garage/reduced_larger_dataset_gated_attention/tfpp_0010_0/config.json \
    --seed 0