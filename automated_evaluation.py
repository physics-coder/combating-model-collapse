import subprocess
import time

# Define the list of model folders
model_folders = ["50100"]

# Define the base path for models
model_base_path = "/home/markgolbraikh/models"
webtext_valid_file = "/home/markgolbraikh/data_train/webtext.valid.json"
synth_valid_file = "/home/markgolbraikh/data_train/small-117M.valid.json"  # Corrected path

# Define the log file
log_file = "evalrun_namedlog.log"

# Function to log messages
def log(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Evaluate each model on both datasets
for folder_name in model_folders:
    model_path = f"{model_base_path}/{folder_name}"

    # Command for evaluating on webtext validation set
    eval_webtext_command = (
        f"python /home/markgolbraikh/transformers/examples/pytorch/language-modeling/run_old.py "
        f"--model_name_or_path {model_path} "
        f"--validation_file {webtext_valid_file} "
        f"--do_eval "
        f"--per_device_eval_batch_size 8 "
        f"--output_dir {model_path}/eval_webtext"
    )

    log(f"Running webtext evaluation command: {eval_webtext_command}")
    subprocess.run(eval_webtext_command, shell=True)

    # Wait for 15 seconds before the next command
    time.sleep(15)

    # Command for evaluating on synthetic validation set
    eval_synth_command = (
        f"python /home/markgolbraikh/transformers/examples/pytorch/language-modeling/run_old.py "
        f"--model_name_or_path {model_path} "
        f"--validation_file {synth_valid_file} "
        f"--do_eval "
        f"--per_device_eval_batch_size 8 "
        f"--output_dir {model_path}/eval_synth"
    )

    log(f"Running synthetic evaluation command: {eval_synth_command}")
    subprocess.run(eval_synth_command, shell=True)

    # Wait for 15 seconds before evaluating the next model
    time.sleep(15)
