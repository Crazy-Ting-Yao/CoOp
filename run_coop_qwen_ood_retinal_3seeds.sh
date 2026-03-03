#!/bin/bash
#SBATCH --job-name=CoOpQwen-OOD-Retinal-3s
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --account=MST114475
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --output=/work/u8686038/log/CoOpQwen-OOD-Retinal-3seeds_%j.log
#SBATCH --error=/work/u8686038/log/CoOpQwen-OOD-Retinal-3seeds_%j.log

echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

ml purge 2>/dev/null || true
ml load miniconda3/24.11.1
ml load cuda/12.4

eval "$(conda shell.bash hook)"
conda activate /home/u8686038/.conda/envs/prograd_qwen

echo "Python: $(which python)"
python -c "import dassl; print('dassl OK:', dassl.__file__)"

export HF_HOME=/work/u8686038/.cache/huggingface
export TRANSFORMERS_CACHE=/work/u8686038/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

cd /work/u8686038/CoOp

DATA_DIR=/work/u8686038/data
TRAINER=CoOpQwen
DATASET=ood_retinal
CFG=qwen2_5_vl
NCTX=16
CSC=False
CTP=end
SHOTS=8

echo "=========================================="
echo "CoOpQwen - OOD Retinal 8-shot (3 seeds)"
echo "Dataset: andyqmongo/IVL_OOD_retinal (8_shot)"
echo "GPUs:    4x H100 80GB"
echo "=========================================="

for SEED in 1 2 3; do
    echo ""
    echo "--- Seed ${SEED} ---"
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Removing old results at ${DIR}"
        rm -rf "$DIR"
    fi
    python train.py \
        --root ${DATA_DIR} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
done

echo ""
echo "=========================================="
echo "All 3 seeds done."
echo "End time: $(date)"
