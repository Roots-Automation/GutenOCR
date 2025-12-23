#!/usr/bin/env python
"""Standardized OCR/VL SFT script with WebDataset streaming and --dry-run.
"""
from __future__ import annotations

import os
import sys
import glob
import io
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Standard logging -------------------------------------------------------
import logging

LOGGER = logging.getLogger("sft")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

# --- Hugging Face / Torch ---------------------------------------------------
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch.utils.data import DataLoader
from accelerate.data_loader import prepare_data_loader as accel_prepare
from accelerate import PartialState
from tqdm.auto import tqdm

# --- WebDataset -------------------------------------------------------------
import webdataset as wds

# --- Project-local imports --------------------------------------------------
from args import parse_sft_args
from prompt_builder import generate_prompt

# --- Constants --------------------------------------------------------------
IGNORE_INDEX = -100
with open("system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_rgb(img: Image.Image) -> Image.Image:
    if not isinstance(img, Image.Image):
        raise TypeError("Expected PIL.Image.Image")
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def get_world_size_and_rank():
    try:
        # Use accelerate's PartialState for proper distributed detection
        state = PartialState()
        return state.num_processes, state.process_index
    except Exception:
        # Fallback to torch.distributed if PartialState fails
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return torch.cuda.device_count(), 0

def is_main_process() -> bool:
    try:
        # Use accelerate's PartialState for proper main process detection
        state = PartialState()
        return state.is_main_process
    except Exception:
        # Fallback to torch.distributed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        # Last resort fallback
        return True


# ------------------ WebDataset streaming -----------------------------------

def _expand_shards(pattern: str) -> List[str]:
    """Expand a user-provided pattern into a list of shard files.

    Supports comma/semicolon/colon-separated lists of globs.
    """
    parts = [p.strip() for p in re.split(r"[,;:]\s*", pattern) if p.strip()]
    files: List[str] = []
    for part in parts:
        expanded = os.path.expanduser(part)
        matches = sorted(p for p in glob.glob(expanded) if Path(p).is_file())
        files.extend(matches)
    if not files:
        raise FileNotFoundError(f"No shards matched: {pattern}")
    return files


def _remap_and_parse(sample):
    # Find JSON data - look for any key ending with .json
    meta = None
    for key, value in sample.items():
        if key.endswith('.json'):
            meta = value
            break
    
    # Fallback to simple 'json' key
    if meta is None and "json" in sample:
        meta = sample["json"]
    
    if meta is None:
        raise KeyError(f"Sample missing JSON data: expected key ending with '.json' or 'json', got keys: {list(sample.keys())}")
    
    if isinstance(meta, (bytes, bytearray)):
        meta = json.loads(meta)
    
    # Find image data - look for any key ending with image extensions
    image_bytes = None
    for key, value in sample.items():
        if key.endswith('.jpg') or key.endswith('.jpeg'):
            image_bytes = value
            break
        elif key.endswith('.png'):
            image_bytes = value
            break
    
    # Fallback to simple extension keys
    if image_bytes is None:
        if "jpg" in sample:
            image_bytes = sample["jpg"]
        elif "jpeg" in sample:
            image_bytes = sample["jpeg"]
        elif "png" in sample:
            image_bytes = sample["png"]
    
    if image_bytes is None:
        raise KeyError(f"Sample missing image data: expected key ending with '.jpg', '.jpeg', '.png', or simple 'jpg'/'png', got keys: {list(sample.keys())}")
    
    return {
        "image_bytes": image_bytes,
        "json_data": meta,
        "sample_id": sample.get("__key__", ""),
    }


def round_down_multiple(x: int, m: int) -> int:
    return x - (x % max(1, m))


def build_wds_iterable(
    patterns, *, per_proc_epoch_size: int, shuffle_shards: int = 1000, shuffle_samples: int = 2000, seed: int = 42, resampled: bool = True,
):
    if isinstance(patterns, str):
        patterns = [patterns]
    all_urls: List[str] = []
    for pat in patterns:
        all_urls.extend(_expand_shards(pat))
    all_urls = sorted(all_urls)

    ds = (
        wds.WebDataset(
            all_urls,
            resampled=resampled,
            shardshuffle=shuffle_shards,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            seed=seed,
            empty_check=False,
        )
        .shuffle(shuffle_samples)
        .decode()
        .map(_remap_and_parse, handler=wds.ignore_and_continue)
        .with_epoch(per_proc_epoch_size)
    )
    return ds


# ------------------ Collator ------------------------------------------------

@dataclass
class QwenVLCollator:
    processor: Any
    max_length: int
    ignore_index: int = IGNORE_INDEX
    allowed_tasks: Optional[List[str]] = None
    allowed_output_types: Optional[List[str]] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images, full_texts, prompt_texts = [], [], []
        for ex in features:
            # Decode image from whichever representation is present
            if "image_bytes" in ex:
                img = _ensure_rgb(Image.open(io.BytesIO(ex["image_bytes"])))
            elif "image" in ex:
                img = _ensure_rgb(ex["image"])  # already PIL
            elif "image_path" in ex:
                with Image.open(ex["image_path"]) as im:
                    img = _ensure_rgb(im.copy())
            else:
                raise KeyError("Example missing image payload: expected image_bytes/image/image_path")

            json_data = ex["json_data"]
            user_prompt, assistant_response = generate_prompt(
                json_data, allowed_tasks=self.allowed_tasks, allowed_output_types=self.allowed_output_types
            )
            if isinstance(assistant_response, (dict, list)):
                assistant_response = json.dumps(assistant_response)
            elif not isinstance(assistant_response, str):
                assistant_response = str(assistant_response)

            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]},
            ]
            prompt_only = conversation[:-1]

            full_texts.append(self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False))
            prompt_texts.append(self.processor.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True))
            images.append(img)

        # Tokenize with the SAME processor/images/max_length for BOTH batches
        full_batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = full_batch["input_ids"].clone()
        # Mask pads
        labels[full_batch["attention_mask"] == 0] = self.ignore_index

        # Mask the entire prompt (including image placeholder tokens)
        prompt_lens = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for i, L in enumerate(prompt_lens):
            L = int(L)
            # If prompt consumes everything, mask everything (no targets)
            if L >= full_batch["input_ids"].size(1) or L >= int(full_batch["attention_mask"][i].sum()):
                labels[i, :] = self.ignore_index
            else:
                labels[i, :L] = self.ignore_index

        full_batch["labels"] = labels
        return full_batch


# ------------------ Trainer with iterable-safe dataloader -------------------

class NoDispatchTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset")
        train_sampler = self._get_train_sampler()
        dl = DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            #prefetch_factor=(self.args.dataloader_prefetch_factor if self.args.dataloader_prefetch_factor > 0 else None), NOTE: no prefetching for stability
            #persistent_workers=self.args.dataloader_persistent_workers, NOTE: without prefetch, this arg can't be used
        )
        st = getattr(self.accelerator, "state", None)
        return accel_prepare(
            dl,
            device=self.accelerator.device,
            num_processes=(st.num_processes if st else None),
            process_index=(st.process_index if st else None),
            split_batches=False,
            dispatch_batches=False,  # crucial for IterableDataset
            put_on_device=False,
            non_blocking=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset")
        dl = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            # prefetch_factor=None,
            # persistent_workers=self.args.dataloader_persistent_workers,
        )
        st = getattr(self.accelerator, "state", None)
        return accel_prepare(
            dl,
            device=self.accelerator.device,
            num_processes=(st.num_processes if st else None),
            process_index=(st.process_index if st else None),
            split_batches=False,
            dispatch_batches=False,
            put_on_device=False,
            non_blocking=True,
        )

# ------------------ Utility: save run config --------------------------------

def _save_run_config(args) -> None:
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        # strip non-serializable bits
        serializable = {k: (str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v)
                        for k, v in vars(args).items()}
        with open(os.path.join(args.output_dir, "run_args.json"), "w") as f:
            json.dump(serializable, f, indent=2, sort_keys=True)
    except Exception as e:
        if is_main_process():
            LOGGER.warning("Failed to save run_args.json: %s", e)

# ------------------ Dry-run helpers -----------------------------------------

def _format_preview(text: str, n: int = 1024) -> str:
    t = text.replace("\n", " ")
    return (t[:n] + ("…" if len(t) > n else "")) if t else ""


def dry_run_preview(args, processor, dataset_iter) -> None:
    """Sample N examples, print prompts/targets, and optionally run generate()."""
    n = getattr(args, "dry_run_samples", 32) or 32

    # We don't need the full collator for printing prompts; but for generation we do.
    collator = QwenVLCollator(
        processor=processor,
        max_length=args.max_length,
        allowed_tasks=args.tasks,
        allowed_output_types=args.output_types,
    )

    picked: List[Dict[str, Any]] = []
    it = iter(dataset_iter)
    # Just take the first N yielded (stream is already shuffled); robust to StopIteration
    while len(picked) < n:
        try:
            ex = next(it)
            picked.append(ex)
        except StopIteration:
            break

    if not picked:
        LOGGER.error("Dry-run found no samples. Check --tar-pattern or dataset setup.")
        return

    # Build minimal per-sample prompt text (and optional generation inputs)
    images: List[Image.Image] = []
    prompt_texts: List[str] = []
    targets: List[str] = []
    for ex in picked:
        # image
        if "image_bytes" in ex:
            img = _ensure_rgb(Image.open(io.BytesIO(ex["image_bytes"])))
        elif "image" in ex:
            img = _ensure_rgb(ex["image"])  # already PIL
        elif "image_path" in ex:
            with Image.open(ex["image_path"]) as im:
                img = _ensure_rgb(im.copy())
        else:
            continue

        # prompts
        user_prompt, assistant_response = generate_prompt(
            ex["json_data"], allowed_tasks=args.tasks, allowed_output_types=args.output_types
        )
        if isinstance(assistant_response, (dict, list)):
            assistant_response = json.dumps(assistant_response)
        elif not isinstance(assistant_response, str):
            assistant_response = str(assistant_response)

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
        ]
        prompt_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        images.append(img)
        prompt_texts.append(prompt_text)
        targets.append(assistant_response)

    # Pretty print preview table (main process only)
    if is_main_process():
        LOGGER.info("—— DRY‑RUN PREVIEW (showing up to %d) ——", len(prompt_texts))
        for i, (pt, tgt, ex) in enumerate(zip(prompt_texts, targets, picked)):
            sid = ex.get("sample_id", "")
            LOGGER.info("[%02d] sample_id=%s\n  prompt: %s\n  target: %s", i + 1, sid, _format_preview(pt), _format_preview(tgt))

    if not getattr(args, "dry_run_generate", False):
        return

    # Optional tiny generation to sanity check wiring
    if getattr(args, "no_model", False):
        LOGGER.warning("--dry-run-generate requested but --no-model is set; skipping generation.")
        return

    LOGGER.info("Running tiny generate() on %d samples …", len(prompt_texts))
    try:
        model_path = args.resume_from_checkpoint or args.model_name
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()
    except Exception as e:
        LOGGER.error("Failed to load model for dry-run generate: %s", e)
        return

    gens_to_run = min(len(prompt_texts), 8)  # keep it small
    inputs = processor(
        text=prompt_texts[:gens_to_run],
        images=images[:gens_to_run],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=getattr(args, "dry_run_max_new_tokens", 24) or 24,
            do_sample=False,
            num_beams=1,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Decode just the new tokens after the input length
    in_len = inputs["input_ids"].shape[1]
    decoded = processor.tokenizer.batch_decode(gen_out[:, in_len:], skip_special_tokens=True)
    for i, txt in enumerate(decoded):
        if is_main_process():
            LOGGER.info("GEN[%02d]: %s", i + 1, _format_preview(txt, n=300))

# ------------------ Main ----------------------------------------------------

def main() -> None:
    args = parse_sft_args()

    # Ensure logging noisiness is controlled per-rank
    if not is_main_process():
        LOGGER.setLevel(logging.WARNING)

    # Validate CUDA and get world size
    if torch.cuda.is_available():
        WORLD_SIZE, RANK = get_world_size_and_rank()
        if is_main_process():
            LOGGER.info("World size: %d (ranks), visible gpus per rank: %d", WORLD_SIZE, torch.cuda.device_count())
    else:
        LOGGER.error("No CUDA GPUs available!")
        return

    # Validate/inspect checkpoint if provided (model-only or full trainer state)
    if args.resume_from_checkpoint:
        checkpoint_path = os.path.abspath(args.resume_from_checkpoint)
        if not os.path.exists(checkpoint_path):
            LOGGER.error("Checkpoint directory not found: %s", checkpoint_path)
            return
        files = set(os.listdir(checkpoint_path))
        missing = []
        if "config.json" not in files:
            missing.append("config.json")
        has_weights = any(
            n in files for n in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"]
        )
        if not has_weights:
            missing.append("model weights (safetensors/bin or sharded)")
        if missing:
            if is_main_process():
                LOGGER.error("Missing critical files in checkpoint: %s", missing)
            return
        if is_main_process():
            LOGGER.info("Checkpoint looks valid: %s", checkpoint_path)

    set_seed(args.seed)

    # Processor is needed for both training and dry-run
    processor_path = args.resume_from_checkpoint or args.model_name
    if is_main_process():
        LOGGER.info("Loading processor from: %s", processor_path)
    processor = AutoProcessor.from_pretrained(processor_path)
    tok = processor.tokenizer
    tok.padding_side = "right"
    tok.truncation_side = "right"

    # ---------------- dataset(s) ----------------
    if is_main_process():
        LOGGER.info("Building WebDataset stream from: %s", args.tar_pattern)
    per_device_bsz = args.batch_size
    grad_accum = args.grad_accum
    
    # Calculate based on true cardinality to hit each sample exactly once
    num_tars = len(_expand_shards(args.tar_pattern))
    N = num_tars * 2048   # Assumption: 2048 samples per tar shard
    global_batch = per_device_bsz * grad_accum * WORLD_SIZE
    
    # Ensure global_batch divides N evenly to avoid drops
    if N % global_batch != 0:
        raise ValueError(
            f"Total samples N={N} must be divisible by global_batch={global_batch} "
            f"(batch_size={per_device_bsz} * grad_accum={grad_accum} * world_size={WORLD_SIZE}) "
            f"to avoid dropping samples. Adjust batch_size or grad_accum."
        )
    
    steps_per_epoch = N // global_batch                      # exact from N
    per_proc_epoch_size = steps_per_epoch * args.batch_size * args.grad_accum
    
    if RANK == 0:
        LOGGER.info("Samples: %d, global_batch: %d, steps/epoch: %d, per-rank epoch size: %d",
                    N, global_batch, steps_per_epoch, per_proc_epoch_size)

    assert per_proc_epoch_size % (args.batch_size * args.grad_accum) == 0

    train_ds = build_wds_iterable(
        patterns=args.tar_pattern,
        resampled=False,  # Critical: no resampling to hit each sample exactly once
        per_proc_epoch_size=per_proc_epoch_size,
        shuffle_shards=num_tars,  # Better mixing across all shards
        shuffle_samples=1000,  # Shuffle within shard
        seed=args.seed,
    )

    # Optional eval stream
    if not args.no_eval and args.val_samples_per_epoch > 0:
        val_total = max(0, int(args.val_samples_per_epoch))
        val_per_proc = max(round_down_multiple(val_total // WORLD_SIZE, per_device_bsz), 1)
        # Use separate eval tar pattern if provided, otherwise use training pattern
        eval_pattern = args.eval_tar_pattern if args.eval_tar_pattern else args.tar_pattern
        val_ds = build_wds_iterable(
            patterns=eval_pattern,
            resampled=False,  # Deterministic eval - no resampling
            per_proc_epoch_size=val_per_proc,
            shuffle_shards=0,  # Minimal shuffle for eval
            shuffle_samples=0,  # Small shuffle buffer for eval variety
            seed=args.seed,
        ) if val_total > 0 else None
    else:
        val_ds = None

    max_steps = int(steps_per_epoch * args.epochs)           # explicitly matches the above, ensure integer
    logging_steps = 10

    # Early exit: DRY RUN -----------------------------------------------------
    if getattr(args, "dry_run", False):
        if is_main_process():
            LOGGER.info("\n===== DRY‑RUN MODE =====")
        if getattr(args, "no_model", False):
            # Only need processor + dataset
            dry_run_preview(args, processor, train_ds)
            return
        else:
            # Load model for optional generation
            model_path = args.resume_from_checkpoint or args.model_name
            if is_main_process():
                LOGGER.info("Loading model from: %s", model_path)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
                attn_implementation="flash_attention_2",
                device_map="auto",
            ).eval()
            # (We only need model handle inside dry_run_preview if generate is requested)
            # The preview function will load its own model to keep scope simple.
            dry_run_preview(args, processor, train_ds)
            return

    # Continue with full training --------------------------------------------
    model_path = args.resume_from_checkpoint or args.model_name
    if is_main_process():
        LOGGER.info("Loading model from: %s", model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        attn_implementation="flash_attention_2",
    )

    # Tokenizer bookkeeping
    tok = processor.tokenizer
    LOGGER.info("emb_vocab_size: %s", model.get_input_embeddings().num_embeddings)
    LOGGER.info(
        "tokenizer_len=%d pad/eos/bos=%s/%s/%s PAD token=%r PAD==EOS? %s",
        len(tok), tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.pad_token, tok.pad_token_id == tok.eos_token_id
    )

    # Freeze modules per your original baseline
    def _freeze_module(mod):
        if mod is None:
            return 0
        n = 0
        for p in mod.parameters():
            if p.requires_grad:
                p.requires_grad = False
                n += p.numel()
        return n

    def _find_first(obj, names):
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n), n
        return None, None

    tok_emb = model.get_input_embeddings()
    frozen_e = _freeze_module(tok_emb)
    LOGGER.info("[freeze] token embeddings: %.2fM params", frozen_e / 1e6)
    lm_head = getattr(model, "lm_head", None)
    frozen_h = _freeze_module(lm_head) if lm_head is not None else 0
    if lm_head is not None:
        tied = (getattr(lm_head, "weight", None) is getattr(tok_emb, "weight", None))
        LOGGER.info("[freeze] lm_head (tied=%s): %.2fM params", tied, frozen_h / 1e6)


    if not args.train_vision:
        vision_mod, vision_name = _find_first(model, ["vision_tower", "vision_model", "visual", "vision"]) or (None, None)
        if vision_mod is None:
            base = getattr(model, "model", None)
            vision_mod, vision_name = _find_first(base or model, ["vision_tower", "vision_model", "visual", "vision"]) or (None, None)
        frozen_v = _freeze_module(vision_mod)
        LOGGER.info("[freeze] vision module '%s': %.2fM params", vision_name, frozen_v / 1e6)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("[freeze] trainable: %.1fM / %.1fM (%.2f%%)", trainable / 1e6, total / 1e6, 100 * trainable / total)

    collator = QwenVLCollator(
        processor=processor,
        max_length=args.max_length,
        allowed_tasks=args.tasks,
        allowed_output_types=args.output_types,
    )

    # TrainingArguments
    overwrite_output_dir = True
    if args.resume_from_checkpoint:
        checkpoint_parent = os.path.dirname(os.path.abspath(args.resume_from_checkpoint))
        output_parent = os.path.abspath(args.output_dir)
        if checkpoint_parent == output_parent:
            overwrite_output_dir = False
            if is_main_process():
                LOGGER.info("Resuming from checkpoint in same output dir; won't overwrite: %s", args.output_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=overwrite_output_dir,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="no" if args.no_eval else "epoch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_only_model=True,
        report_to=["tensorboard"],
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.persistent_workers, # defaults false for stability
        # dataloader_prefetch_factor=args.prefetch_factor, NOTE: no prefetching for stability
        dataloader_drop_last=True,
        max_grad_norm=1.0,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        tf32=True,
        optim="adamw_torch_fused",
        deepspeed=getattr(args, "deepspeed_config", None),
        eval_on_start=True
    )

    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    _save_run_config(args)

    if is_main_process():
        LOGGER.info("Initializing Trainer … eval=%s", ("off" if args.no_eval else "epoch"))
    trainer = NoDispatchTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None if args.no_eval else val_ds,
        data_collator=collator,
        processing_class=processor,
    )

    # Determine resume mode
    resume_from_checkpoint = None
    if args.resume_from_checkpoint and not args.load_model_only:
        trainer_state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            resume_from_checkpoint = args.resume_from_checkpoint
            if is_main_process():
                LOGGER.info("Will resume training state from: %s", resume_from_checkpoint)
        else:
            if is_main_process():
                LOGGER.info("No trainer_state.json; starting fresh training state")
    elif args.load_model_only:
        if is_main_process():
            LOGGER.info("Loading model weights only; starting fresh training state")

    if is_main_process():
        LOGGER.info("Starting training …")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if not args.no_eval:
        if is_main_process():
            LOGGER.info("Evaluating …")
        metrics = trainer.evaluate()
        if is_main_process():
            for k, v in metrics.items():
                LOGGER.info("%s: %s", k, v)

    if is_main_process():
        LOGGER.info("Saving model + processor to %s …", args.output_dir)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    if is_main_process():
        LOGGER.info("Done.")


if __name__ == "__main__":
    main()
