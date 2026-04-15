import subprocess
import time
import glob
import re
from pathlib import Path
import random
import os

# Configuration
code_root = r"/home/mohamednaas/Documents/carla_garage"
carla_root = r"/home/mohamednaas/Documents/carla_garage/carla"
data_save_directory = r"/home/mohamednaas/Documents/carla_project/dataset_small_model"
route_folder = r"/home/mohamednaas/Documents/carla_garage/data"

# Set fixed ports
FREE_STREAMING_PORT = 10010
FREE_WORLD_PORT = 20010
TM_PORT = 30010

# Constants for partial extraction
N_SCENARIOS = 1000
COLLECTED_SCENARIOS = 0

# Find all routes
routes = glob.glob(f"{route_folder}/**/*.xml", recursive=True)

# Create directories
Path(f"{data_save_directory}/start_files").mkdir(parents=True, exist_ok=True)
# Export important paths into this process environment so subprocesses inherit them
os.environ.setdefault('SCENARIO_RUNNER_ROOT', f"{code_root}/scenario_runner_autopilot")
os.environ.setdefault('LEADERBOARD_ROOT', f"{code_root}/leaderboard_autopilot")
os.environ.setdefault('CARLA_ROOT', f"{carla_root}")
os.environ.setdefault('CARLA_SERVER', f"{carla_root}/CarlaUE4.sh")
os.environ.setdefault('PYTHONPATH', os.environ.get('PYTHONPATH', '') + f":{carla_root}/PythonAPI/carla:{code_root}/leaderboard_autopilot:{code_root}/scenario_runner_autopilot")
random.seed(42)

n = 0
s = 0
for route in random.sample(routes, k=N_SCENARIOS)[COLLECTED_SCENARIOS:]:

    # Extract route info
    routefile_number = route.split("/")[-1].split(".")[0]
    town = re.search('Town(\\d+)', route).group(0)
    scenario_type = route.split("/")[-2]
    s += 1
    # Set up paths
    save_path = f"{data_save_directory}/data/{scenario_type}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    ckpt_endpoint = f"{data_save_directory}/results/{scenario_type}/{routefile_number}_result.json"
    agent = f"{code_root}/team_code/data_agent.py"
    
    world_port = FREE_WORLD_PORT + n
    streaming_port = FREE_STREAMING_PORT + n
    tm_port = TM_PORT + n
    # Create bash script
    jobfile = f"{data_save_directory}/start_files/{routefile_number}.sh"
    qsub_template = f"""#!/bin/bash
                    # Export environment for child process
                    export SCENARIO_RUNNER_ROOT={code_root}/scenario_runner_autopilot
                    export LEADERBOARD_ROOT={code_root}/leaderboard_autopilot
                    export CARLA_ROOT={carla_root}
                    export CARLA_SERVER={carla_root}/CarlaUE4.sh
                    export PYTHONPATH="$PYTHONPATH:{carla_root}/PythonAPI/carla:{code_root}/leaderboard_autopilot:{code_root}/scenario_runner_autopilot"
                    export REPETITIONS=1
                    export REPETITION=0
                    export DEBUG_CHALLENGE=0
                    export TEAM_AGENT="{agent}"
                    export CHALLENGE_TRACK_CODENAME=MAP
                    export ROUTES="{route}"
                    export TOWN="{town}"
                    export TM_SEED={s}
                    export CHECKPOINT_ENDPOINT="{ckpt_endpoint}"
                    export TEAM_CONFIG=/home/mohamednaas/Documents/carla_garage/reduced_model_longer_training_training_out/tfpp_0010_0
                    export RESUME=1
                    export DATAGEN=1
                    export SAVE_PATH="{save_path}"

                    echo "Start python"

                    export FREE_STREAMING_PORT={streaming_port}
                    export FREE_WORLD_PORT={world_port}
                    export TM_PORT={tm_port}

                    echo "FREE_STREAMING_PORT: $FREE_STREAMING_PORT"
                    echo "FREE_WORLD_PORT: $FREE_WORLD_PORT"
                    echo "TM_PORT: $TM_PORT"

                    # Start CARLA in background
                    "$CARLA_SERVER" -carla-world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &

                    sleep 5

                    "/home/mohamednaas/miniconda3/envs/garage_2/bin/python" "{code_root}/leaderboard/leaderboard/leaderboard_evaluator_local.py" --host=127.0.0.1 --port=$FREE_WORLD_PORT \
                        --traffic-manager-port=$TM_PORT --traffic-manager-seed=$TM_SEED --routes="$ROUTES" --repetitions=$REPETITIONS \
                        --track=$CHALLENGE_TRACK_CODENAME --checkpoint="$CHECKPOINT_ENDPOINT" --agent="$TEAM_AGENT" \
                        --agent-config="$TEAM_CONFIG" --debug=0 --resume=$RESUME --timeout=600
"""

    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    
    # Run the job
    print(f"Running job for route {routefile_number}")
    result = subprocess.run(f"bash {jobfile}", shell=True)
    
    # Wait for completion
    if result.returncode != 0:
        print(f"Job failed for route {routefile_number}")
    time.sleep(5)  # Small delay between jobs