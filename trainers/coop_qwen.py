import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer


def load_qwen_to_cpu(cfg):
    model_name = cfg.MODEL.BACKBONE.NAME
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model = model.cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def images_to_qwen_pixels(images, temporal_patch_size=2, patch_size=14):
    """Convert [B, C, H, W] tensors to Qwen2.5-VL vision encoder format."""
    B, C, H, W = images.shape
    hp = H // patch_size
    wp = W // patch_size
    imgs = images.unsqueeze(2).expand(-1, -1, temporal_patch_size, -1, -1).contiguous()
    imgs = imgs.reshape(B, C, temporal_patch_size, hp, patch_size, wp, patch_size)
    imgs = imgs.permute(0, 3, 5, 1, 2, 4, 6)
    pixel_values = imgs.reshape(B * hp * wp, C, temporal_patch_size, patch_size, patch_size)
    grid_thw = torch.tensor(
        [[1, hp, wp]] * B, dtype=torch.long, device=images.device
    )
    return pixel_values, grid_thw


class QwenTextEncoder(nn.Module):

    def __init__(self, qwen_model):
        super().__init__()
        self.transformer = qwen_model.model.language_model
        self.hidden_size = qwen_model.config.hidden_size

    def forward(self, inputs_embeds, attention_mask):
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = outputs.last_hidden_state
        last_idx = attention_mask.sum(dim=1).long() - 1
        text_features = hidden_states[
            torch.arange(batch_size, device=device), last_idx
        ]
        return text_features


class QwenPromptLearner(nn.Module):

    def __init__(self, cfg, classnames, qwen_model, tokenizer):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_dim = qwen_model.config.hidden_size
        dtype = next(qwen_model.parameters()).dtype
        ctx_init = cfg.TRAINER.COOP.CTX_INIT

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            ctx_init_ids = tokenizer.encode(ctx_init, add_special_tokens=False)
            n_ctx = len(ctx_init_ids)
            with torch.no_grad():
                ctx_init_tensor = torch.tensor([ctx_init_ids], dtype=torch.long)
                ctx_vectors = qwen_model.model.language_model.embed_tokens(
                    ctx_init_tensor
                ).squeeze(0).to(dtype)
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [
            len(tokenizer.encode(name, add_special_tokens=False))
            for name in classnames
        ]
        suffixes = [name + "." for name in classnames]
        suffix_tokens_list = [
            tokenizer.encode(s, add_special_tokens=False) for s in suffixes
        ]
        max_suffix_len = max(len(t) for t in suffix_tokens_list)

        padded_suffix_ids = torch.zeros(n_cls, max_suffix_len, dtype=torch.long)
        suffix_mask = torch.zeros(n_cls, max_suffix_len, dtype=torch.long)
        for i, tokens in enumerate(suffix_tokens_list):
            padded_suffix_ids[i, : len(tokens)] = torch.tensor(tokens)
            suffix_mask[i, : len(tokens)] = 1

        with torch.no_grad():
            suffix_embeddings = qwen_model.model.language_model.embed_tokens(
                padded_suffix_ids
            ).to(dtype)

        self.register_buffer("suffix_embeddings", suffix_embeddings)
        self.register_buffer("suffix_mask", suffix_mask)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.cat([ctx, self.suffix_embeddings], dim=1)
        ctx_mask = torch.ones(
            self.n_cls, self.n_ctx,
            device=ctx.device, dtype=self.suffix_mask.dtype,
        )
        attention_mask = torch.cat([ctx_mask, self.suffix_mask], dim=1)
        return prompts, attention_mask


class CustomQwenVL(nn.Module):

    def __init__(self, cfg, classnames, qwen_model, tokenizer):
        super().__init__()
        self.prompt_learner = QwenPromptLearner(
            cfg, classnames, qwen_model, tokenizer
        )
        self.image_encoder = qwen_model.model.visual
        self.text_encoder = QwenTextEncoder(qwen_model)
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
        self.dtype = next(qwen_model.parameters()).dtype

        vc = qwen_model.config.vision_config
        self.patch_size = vc.patch_size
        self.temporal_patch_size = vc.temporal_patch_size
        self.spatial_merge_size = vc.spatial_merge_size
        self.hidden_size = qwen_model.config.hidden_size

    def encode_image(self, image):
        B = image.shape[0]
        pixel_values, grid_thw = images_to_qwen_pixels(
            image.to(self.dtype),
            temporal_patch_size=self.temporal_patch_size,
            patch_size=self.patch_size,
        )
        visual_features = self.image_encoder(pixel_values, grid_thw=grid_thw)
        merge = self.spatial_merge_size
        merged_h = grid_thw[0, 1].item() // merge
        merged_w = grid_thw[0, 2].item() // merge
        t_patches = grid_thw[0, 0].item()
        n_merged = t_patches * merged_h * merged_w
        image_features = visual_features.reshape(B, n_merged, -1).mean(dim=1)
        return image_features

    def forward(self, image):
        image_features = self.encode_image(image)
        prompts, attention_mask = self.prompt_learner()
        text_features = self.text_encoder(prompts, attention_mask)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class CoOpQwen(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp", "bf16"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading Qwen2.5-VL (backbone: {cfg.MODEL.BACKBONE.NAME})")
        qwen_model, tokenizer = load_qwen_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32":
            qwen_model.float()

        print("Building custom Qwen-VL model")
        self.model = CustomQwenVL(cfg, classnames, qwen_model, tokenizer)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "logit_scale" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(
                self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS
            )

        self.model.to(self.device)

        trainable_params = [
            {"params": self.model.prompt_learner.parameters()},
            {"params": [self.model.logit_scale], "lr": cfg.OPTIM.LR * 0.1},
        ]
        self.optim = build_optimizer(trainable_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner", self.model.prompt_learner,
            self.optim, self.sched,
        )

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "suffix_embeddings" in state_dict:
                del state_dict["suffix_embeddings"]
            if "suffix_mask" in state_dict:
                del state_dict["suffix_mask"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict, strict=False)
