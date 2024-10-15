# setup environment variables
export SHELL_PATH=$(dirname $(readlink -f $0))
export WORKSPACE=${SHELL_PATH}/..
export CARLA_ROOT=${WORKSPACE}/CARLA_Leaderboard_20

export AUTOPILOT_ROOT=${WORKSPACE}/carla_autopilot
export LEADERBOARD_ROOT=${AUTOPILOT_ROOT}/leaderboard_20/leaderboard
export SCENARIO_RUNNER_ROOT=${AUTOPILOT_ROOT}/leaderboard_20/scenario_runner_custom

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}
export PYTHONPATH=$PYTHONPATH:${WORKSPACE}
export PYTHONPATH=$PYTHONPATH:${AUTOPILOT_ROOT}

# general parameters
PORT=2000
TM_PORT=2500
DEBUG_CHALLENGE=0

export ROUTES=${LEADERBOARD_ROOT}/data/routes_training.xml
export ROUTES_SUBSET=0
REPETITIONS=1

# agent-related options
CHALLENGE_TRACK_CODENAME=MAP
TEAM_AGENT=${WORKSPACE}/tfpp/data_agent_v2.py
TEAM_CONFIG=${AUTOPILOT_ROOT}/carla_autopilot/configs/expert_agent.yaml
RESUME=0
CHECKPOINT_ENDPOINT=${WORKSPACE}/logs/L20_training/log_route_${ROUTES_SUBSET}.json

export DATAGEN=1
export SAVE_PATH=${WORKSPACE}/data/Routes_${ROUTES_SUBSET}_Repetition${REPETITIONS}
export PYTHON_FILE=${AUTOPILOT_ROOT}/carla_autopilot/leaderboard_custom/leaderboard_evaluator.py

python3 ${PYTHON_FILE} \
    --port=${PORT} \
    --traffic-manager-port=${TM_PORT} \
    --debug=${DEBUG_CHALLENGE} \
    --routes=${ROUTES} \
    --routes-subset=${ROUTES_SUBSET} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --resume=${RESUME} \
    --checkpoint=${CHECKPOINT_ENDPOINT}
