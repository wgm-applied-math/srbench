# Common experiment parameters
COMMON_ARGS="../datasets/blackbox/ \
    -script optimize_model \
    -results ../results_blackbox_tuning/ \
    -n_trials 2 -job_time_limit 0:05 -fit_time_limit 60 \
    -max_samples 4 \
    --local \
    --scale_x --scale_y" 

# Submit jobs for CPU-based algorithms
python analyze.py $COMMON_ARGS -ml jessamine

