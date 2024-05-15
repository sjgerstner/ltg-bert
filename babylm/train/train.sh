#!/bin/bash

#SBATCH --job-name=ltgbert
#SBATCH --time=47:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=report/%j.out
#SBATCH --signal=B:TERM

### Note: --gres=gpu:x should equal to ntasks-per-node

args=("$@")

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error


# Load modules
module --quiet purge


# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_VERBOSE=2

export WANDB_MODE=offline

trap 'echo signal recieved in BATCH!; kill -15 "${PID}"; wait "${PID}";' SIGINT SIGTERM

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# export NCCL_DEBUG=INFO

etc_passwd=${HOME}/.local/passwd
if [[ ! -e ${etc_passwd} ]]; then
  getent passwd ${USER} slurm > ${etc_passwd}
fi

grad_accum=1
steps=$((150000 * $grad_accum))

seed=$1


srun singularity exec --nv \
--overlay /scratch/myh2014/singularity/50G.ext3:ro \
--bind /opt/slurm/etc,/var/run/munge,/opt/slurm/etc:/ext3/slurm/etc:ro,${etc_passwd}:/etc/passwd:ro \
/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif  /bin/bash -c \
"source /ext3/env.sh; conda activate tpr; \
python train.py --batch_size 64 --max_steps $steps --input_path ../data/processed_100M/cached_train_128.txt --config_file ../configs/base.json --output_dir ../checkpoints/ltgbert_base_v2_3e3 \
--vocab_path ../tokenizer.json --gradient_accumulation_steps $grad_accum --seed $seed --learning_rate 3e-3 \
--checkpoint_path ../checkpoints/ltgbert_base_v2_3e3/model_6_$seed.bin"  

# python train.py --batch_size 128 --max_steps $steps --input_path ../data/processed_10M/cached_train_128.txt --config_file ../configs/ltgbert_small_elc_tok.json --output_dir ../checkpoints/ltgbert_small_elc_tok --vocab_path ../tokenizer_10M_elcbert.json --gradient_accumulation_steps $grad_accum --seed $seed"

# srun singularity exec --nv \
# --overlay /scratch/myh2014/singularity/50G.ext3:ro \
# --bind /opt/slurm/etc,/var/run/munge,/opt/slurm/etc:/ext3/slurm/etc:ro,${etc_passwd}:/etc/passwd:ro \
# /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif  /bin/bash -c \
# "source /ext3/env.sh; conda activate tpr; \
# python train.py --batch_size 64 --max_steps 31250 --input_path ../data/processed_10M/all.txt --config_file ../configs/ltgbert_small_large_tok.json --output_dir ../checkpoints/ltgbert_small_large_tok --vocab_path ../tokenizer_10M.json"

# export PS1=\$; \
# export NCCL_SOCKET_IFNAME=hsn; \
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; \
# export OPENBLAS_VERBOSE=2; \
# export WANDB_MODE=offline; \
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) \
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE)) \
# export MASTER_ADDR=$master_addr \


# torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_NTASKS_PER_NODE --max_restarts=3 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --master_addr ${MASTER_ADDR} --master_port $MASTER_PORT train.py --batch_size 64 "$@" &

# torchrun --standalone --nnodes=1 --nproc_per_node=${N_GPUS} train.py "$@" &
