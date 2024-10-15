# setup environment variables
export SHELL_PATH=$(dirname $(readlink -f $0))
export WORKSPACE=${SHELL_PATH}/..
export CARLA_ROOT=${WORKSPACE}/CARLA_Leaderboard_10
export LEADERBOARD_ROOT=${WORKSPACE}/leaderboard_10/leaderboard
export SCENARIO_RUNNER_ROOT=${WORKSPACE}/leaderboard_10/scenario_runner

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}
export PYTHONPATH=$PYTHONPATH:${WORKSPACE}

# general parameters
PORT=2000
TM_PORT=2500
DEBUG_CHALLENGE=0

ROUTES=${LEADERBOARD_ROOT}/data/routes_testing.xml
ROUTES_SUBSET=0
SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json
REPETITIONS=1

# agent-related options
CHALLENGE_TRACK_CODENAME=SENSORS
TEAM_AGENT=${WORKSPACE}/team_code/sensor_agent.py
TEAM_CONFIG=${WORKSPACE}/ckpt/lav/tfpp_02_05_withheld_0
RESUME=0
CHECKPOINT_ENDPOINT=${WORKSPACE}/logs/L10_testing/log_route_${ROUTES_SUBSET}.json

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --port=${PORT} \
    --trafficManagerPort=${TM_PORT} \
    --debug=${DEBUG_CHALLENGE} \
    --scenarios=${SCENARIOS}  \
    --routes=${ROUTES} \
    --routes-subset=${ROUTES_SUBSET} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --resume=${RESUME} \
    --checkpoint=${CHECKPOINT_ENDPOINT}
