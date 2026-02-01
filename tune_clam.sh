#!/bin/bash

# Script to tune Model Architecture, Learning Rate, and Weight Decay

# the following parameters can be modified as needed:
model_types=("mil" "clam_sb")
learning_rates=(2e-4 1e-4 5e-5)
weight_decays=(1e-3 1e-4)
# ----------------------------------------

# Create a master log directory for bash outputs
mkdir -p bash_logs

for model in "${model_types[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for reg in "${weight_decays[@]}"; do
      
      
      EXP_NAME="debug_fold_1_${model}_lr${lr}_reg${reg}"
      LOG_FILE="bash_logs/${EXP_NAME}.log"
      
      echo "==================================================" | tee -a "$LOG_FILE"
      echo "Starting experiment: $EXP_NAME at $(date)" | tee -a "$LOG_FILE"
      echo "Settings: Model=${model}, LR=${lr}, Reg=${reg}" | tee -a "$LOG_FILE"
      echo "==================================================" | tee -a "$LOG_FILE"
      
      # 1. Run the Python script
      # 2. '2>&1' redirects errors to the same output
      # 3. 'tee' saves output to file AND shows it on screen
      CUDA_VISIBLE_DEVICES=0 python main.py \
        --data_root_dir '/media/savirlab/My Book/SCC_CLAM/features_data_dir' \
        --drop_out 0.5 \
        --early_stopping \
        --lr "$lr" \
        --k 4 \
        --k_start 1 \
        --k_end 2 \
        --reg "$reg" \
        --max_epochs 50 \
        --weighted_sample \
        --bag_loss ce \
        --inst_loss svm \
        --task task_3_recurrance_prediction \
        --model_type "$model" \
        --exp_code "$EXP_NAME" \
        --log_data \
        --embed_dim 1024 2>&1 | tee -a "$LOG_FILE"

      # Check if the python script failed (exit code not 0)
      if [ $? -ne 0 ]; then
          echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOG_FILE"
          echo "CRITICAL FAILURE in $EXP_NAME. Stopping Loop." | tee -a "$LOG_FILE"
          exit 1
      fi

      echo "Finished $EXP_NAME at $(date)" | tee -a "$LOG_FILE"
      
      # --- SAFETY COOL-DOWN ---
      # Sleep for 30 seconds to avoid overheating GPU
      echo "Cooling down GPU for 30 seconds..."
      sleep 30
        
    done
  done
done