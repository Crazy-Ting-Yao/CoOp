# How to run

The single entry point is `train.py`. You run training/evaluation by selecting a `--trainer` and providing two config files (dataset config / trainer config).

## Setup environment


```bash
# 1) Create and activate an environment (example)
conda create -n coop python=3.10 -y
conda activate coop

# 2) Install PyTorch (pick the correct command for your CUDA version)
# See: https://pytorch.org/get-started/locally/

# 3) Install Dassl.pytorch (follow their installation instructions)
# Repo: https://github.com/KaiyangZhou/Dassl.pytorch

# 4) Install this repo's extra dependencies
pip install -r requirements.txt
```

If you are using HuggingFace-hosted datasets that require authentication, set:

```bash
export HF_TOKEN=your_hf_token
```

## Train (template)

```bash
python train.py \
  --root /path/to/datasets \
  --trainer <TRAINER_NAME> \
  --dataset-config-file configs/datasets/<DATASET>.yaml \
  --config-file configs/trainers/<METHOD>/<CFG>.yaml \
  --output-dir output/<DATASET>/<TRAINER>/<EXP_NAME>
```

Common optional flags:

- Fix the random seed: `--seed 1`
- Override config values by appending key-value pairs at the end, e.g.:
  - `DATASET.NUM_SHOTS 1`
  - `TRAINER.COOP.N_CTX 16`
  - `TRAINER.COOP.CSC False`
  - `TRAINER.COOP.CLASS_TOKEN_POSITION end`

## Evaluate (eval-only)

```bash
python train.py \
  --eval-only \
  --model-dir output/<DATASET>/<TRAINER>/<EXP_NAME> \
  --load-epoch <EPOCH>
```

## Run CoOp / CoCoOp (original CLIP)

For full reproduction commands/recipes, see:

- CoOp: `COOP.md`
- CoCoOp: `COCOOP.md`

(Both are launched via `train.py`; only the trainer/config differs.)

## Run this fork: Qwen2.5-VL / LLaVA

### Qwen2.5-VL (`CoOpQwen`)

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpQwen \
  --dataset-config-file configs/datasets/ood_retinal.yaml \
  --config-file configs/trainers/CoOp/qwen2_5_vl.yaml
```

3-seed script (equivalent to the command above; runs seed=1/2/3):

```bash
bash scripts/coop/main_qwen.sh ood_retinal end 16 1 False
```

### LLaVA v1.6 / LLaVA-NeXT (`CoOpLlava`)

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpLlava \
  --dataset-config-file configs/datasets/retinal.yaml \
  --config-file configs/trainers/CoOp/llava_v16.yaml
```
