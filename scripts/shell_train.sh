# setup environment variables
export SHELL_PATH=$(dirname $(readlink -f $0))
export WORKSPACE=${SHELL_PATH}/..
export CARLA_ROOT=${WORKSPACE}/CARLA_Leaderboard_20

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${WORKSPACE}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/miniconda3/lib

export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.

DATASET=${WORKSPACE}/data
LOGDIR=${WORKSPACE}/runtime

torchrun --nnodes=1 \
    --nproc_per_node=1 \
    --max_restarts=1 \
    --rdzv_id=42353467 \
    --rdzv_backend=c10d ${WORKSPACE}/tfpp/train.py \
    --id train_id_000 \
    --batch_size 6 \
    --setting all \
    --root_dir ${DATASET} \
    --logdir ${LOGDIR} \
    --use_controller_input_prediction 1 \
    --use_wp_gru 0 \
    --use_discrete_command 1 \
    --use_tp 1 \
    --continue_epoch 1 \
    --cpu_cores 20 \
    --num_repetitions 2
