#!/bin/bash

# Script to tune Model Architecture, Learning Rate, and Weight Decay

# the following parameters can be modified as needed:
ecnoding_dim=(256)
attn_dim=(128)
learning_rates=(2e-4)
# ----------------------------------------

# Create a master log directory for bash outputs
mkdir -p bash_logs

for enc in "${ecnoding_dim[@]}"; do
  for attn in "${attn_dim[@]}"; do
    for lr in "${learning_rates[@]}"; do
      
      
      EXP_NAME="debug_ce4_fold_3n4_enc${enc}_attn${attn}_lr${lr}"
      LOG_FILE="bash_logs/${EXP_NAME}.log"
      
      echo "==================================================" | tee -a "$LOG_FILE"
      echo "Starting experiment: $EXP_NAME at $(date)" | tee -a "$LOG_FILE"
      echo "Settings: Encoding Dim=${enc}, Attn Dim=${attn}, LR=${lr}" | tee -a "$LOG_FILE"
      echo "==================================================" | tee -a "$LOG_FILE"
      
      # 1. Run the Python script
      # 2. '2>&1' redirects errors to the same output
      # 3. 'tee' saves output to file AND shows it on screen
      CUDA_VISIBLE_DEVICES=0 python main.py \
        --drop_out 0.5 \
        --early_stopping \
        --lr "$lr" \
        --k 4 \
        --k_start 3 \
        --k_end 5 \
        --reg 1e-4 \
        --max_epochs 100 \
        --weighted_sample \
        --bag_loss ce \
        --no_inst_cluster \
        --task task_3_recurrance_prediction \
        --model_type clam_sb \
        --exp_code "$EXP_NAME" \
        --log_data \
        --embed_dim 1024 \
        --encoding_dim "$enc" \
        --attn_dim "$attn"  2>&1 | tee -a "$LOG_FILE"

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