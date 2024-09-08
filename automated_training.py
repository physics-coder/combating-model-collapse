import subprocess
import time

# Define the log file
log_file = "bigrunlog.log"

# Function to log messages
def log(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Define the list of replace ratios
replace_ratios = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# First loop: Create folders and run the initial command
for ratio in replace_ratios:
    folder_name = f"{int(ratio * 100):02d}00"
    
    command = (
        f"python replace_pro.py data_train/small-117M.train.json "
        f"data_mod/{folder_name} --replace_ratio {ratio} --far_ratio 0.0"
    )
    
    log(f"Running command: {command}")
    subprocess.run(command, shell=True)
    
    time.sleep(30)

# Second loop: Run the training command for each created folder
for ratio in replace_ratios:
    folder_name = f"{int(ratio * 100):02d}00"
    
    # Construct the command for the training process
    training_command = (
        f"python /home/markgolbraikh/transformers/examples/pytorch/xla_spawn.py "
        f"--num_cores 8 "
        f"/home/markgolbraikh/transformers/examples/pytorch/language-modeling/run_clm.py "
        f"--model_type gpt2 "
        f"--config_name /home/markgolbraikh/transformers/examples/pytorch/language-modeling/gpt2_config.json "
        f"--train_file /home/markgolbraikh/data_mod/{folder_name}/output.json "
        f"--do_train "
        f"--num_train_epochs 3 "
        f"--per_device_train_batch_size 16 "
        f"--output_dir /home/markgolbraikh/models/{folder_name} "
        f"--overwrite_output_dir "
        f"--tokenizer_name gpt2 "
        f"--block_size 1024 "
        f"--optim adamw_torch "
        f"--learning_rate 5e-5 "
        f"--weight_decay 0.1 "
        f"--warmup_ratio 0.1 "
        f"--lr_scheduler_type cosine "
        f"--save_strategy epoch "
        f"--logging_strategy no "
        f"--use_fast_tokenizer "
        f"--preprocessing_num_workers 70 "
        f"--dataloader_num_workers 16"
    )
    
    log(f"Running training command: {training_command}")
    subprocess.run(training_command, shell=True)

    # Wait for 15 seconds before clearing the cache
    time.sleep(15)
    
    # Clear the cache
    clear_cache_command = "rm -rf ~/.cache/huggingface/datasets"
    log(f"Clearing cache with command: {clear_cache_command}")
    subprocess.run(clear_cache_command, shell=True)
    
    # Wait for another 15 seconds before the next command
    time.sleep(15)
