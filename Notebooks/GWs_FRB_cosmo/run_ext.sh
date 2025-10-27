#!/bin/bash
#SBATCH --job-name="DM_ext"
#SBATCH --output="%j.DM_ext.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --account=unv116
#SBATCH --ntasks-per-node=96
#SBATCH -t 48:00:00
#SBATCH --mem=15G
#SBATCH --constraint="lustre"

module purge
module load shared
module load slurm
module load cpu/0.17.3b

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310

monitor_memory() {
    local pid=$1
    local interval=1800
    local sample_rate=600
    local max_mem=0
    local max_vms=0
    local sample_count=0
    
    echo "========================================="
    echo "Memory monitoring started at $(date)"
    echo "PID: $pid | Report interval: ${interval}s"
    echo "========================================="
    
    local last_report_time=$(date +%s)
    
    while kill -0 $pid 2>/dev/null; do
        mem_info=$(ps -p $pid -o rss=,vsz= 2>/dev/null)
        if [ -n "$mem_info" ]; then
            rss=$(echo $mem_info | awk '{print $1}')
            vms=$(echo $mem_info | awk '{print $2}')
            rss_mb=$((rss / 1024))
            vms_mb=$((vms / 1024))
            
            if [ $rss_mb -gt $max_mem ]; then
                max_mem=$rss_mb
            fi
            if [ $vms_mb -gt $max_vms ]; then
                max_vms=$vms_mb
            fi
            
            sample_count=$((sample_count + 1))
        fi
        
        current_time=$(date +%s)
        elapsed=$((current_time - last_report_time))
        
        if [ $elapsed -ge $interval ]; then
            echo ""
            echo "--- Memory Report at $(date) ---"
            echo "Current Memory: ${rss_mb} MB (RSS), ${vms_mb} MB (VMS)"
            echo "PEAK Memory:    ${max_mem} MB (RSS), ${max_vms} MB (VMS)"
            echo "Samples taken:  ${sample_count}"
            echo "--------------------------------------"
            last_report_time=$current_time
        fi
        
        sleep $sample_rate
    done

    echo ""
    echo "========================================="
    echo "FINAL Memory Report at $(date)"
    echo "PEAK RSS Memory: ${max_mem} MB"
    echo "PEAK VMS Memory: ${max_vms} MB"
    echo "Total samples:   ${sample_count}"
    echo "========================================="
}

echo "Starting Python script at $(date)"
python3 -W ignore::DeprecationWarning Cosmo_constraints_DM_ext_MCMC.py &
PYTHON_PID=$!

echo "Python PID: $PYTHON_PID"

monitor_memory $PYTHON_PID

wait $PYTHON_PID
EXIT_CODE=$?

echo ""
echo "Python script exited with code: $EXIT_CODE at $(date)"

exit $EXIT_CODE