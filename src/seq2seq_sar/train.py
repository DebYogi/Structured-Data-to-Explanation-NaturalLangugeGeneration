"""Simple training script for fine-tuning a seq2seq model (T5/BART) using HF Transformers.
Deterministic seed and simple training loop suitable for small-scale experiments.
"""
import argparse
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

from .dataset_builder import build_pairs, SARPairDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    # Device detection
    import torch
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if cuda_available else 0

    # Determine effective device and fp16 behavior
    if args.device and args.device.lower() != "auto":
        device = args.device
    else:
        device = "cuda" if cuda_available else "cpu"

    # If user didn't explicitly set fp16, enable it when CUDA is available
    fp16_enabled = args.fp16 if args.fp16 is not None else (True if device == "cuda" else False)

    print(f"Training device: {device} (gpus={num_gpus}) | fp16: {fp16_enabled}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pairs = build_pairs([f"ACC_{i:04d}" for i in range(1, args.n_accounts + 1)], args.year, args.month, seed=args.seed)
    srcs = [p[0] for p in pairs]
    tgts = [p[1] for p in pairs]
    ds = Dataset.from_dict({"src": srcs, "tgt": tgts})

    def tokenize_fn(examples):
        model_inputs = tokenizer(examples["src"], truncation=True, padding='max_length', max_length=args.max_len)
        labels = tokenizer(examples["tgt"], truncation=True, padding='max_length', max_length=args.max_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"])

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Adaptive batch sizing if GPU(s) available and user did not override
    per_device_batch = args.batch_size
    if per_device_batch is None:
        per_device_batch = 8 if num_gpus >= 1 else 2

    if args.use_accelerate:
        try:
            from accelerate import Accelerator
        except Exception as e:
            raise ImportError("`accelerate` is required for --use-accelerate. Install with `pip install accelerate`") from e

        # Prepare dataset for PyTorch DataLoader
        tokenized.set_format(type='torch', columns=[c for c in tokenized.column_names if c in ['input_ids', 'attention_mask', 'labels']])
        import torch
        from torch.utils.data import DataLoader

        dataloader = DataLoader(tokenized, batch_size=per_device_batch, shuffle=True, collate_fn=data_collator)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        accelerator = Accelerator(mixed_precision='fp16' if fp16_enabled else 'no')
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

        model.train()
        for epoch in range(args.epochs):
            for step, batch in enumerate(dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        # Save model
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(args.output_dir)
    else:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            save_total_limit=1,
            fp16=fp16_enabled,
            logging_steps=10,
            save_strategy="epoch",
            dataloader_pin_memory=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Trainer will use CUDA automatically if available and not overridden by environment
        trainer.train()
        trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="t5-small")
    parser.add_argument("--n-accounts", type=int, default=50)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="outputs/sar_model")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--fp16", type=lambda x: (str(x).lower() == 'true'), nargs='?', const=True, default=None, help="Enable/disable fp16 (default: auto-enable on CUDA)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--use-accelerate", action="store_true", dest="use_accelerate", help="Use `accelerate` for distributed / fp16 training with Accelerator")
    args = parser.parse_args()
    main(args)
