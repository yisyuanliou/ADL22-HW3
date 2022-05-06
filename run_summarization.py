import os
import torch
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataset import SummarizeDataset
from datasets import load_metric
from transformers import (
    AutoTokenizer, 
    DataCollatorForSeq2Seq, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

def load_dataset(args):
    data = {}

    data_paths = os.path.join(args.data_dir, "train.jsonl")
    with  open(data_paths, 'r', encoding="utf-8") as json_file:
        json_list = list(json_file)
        data["train"] = [json.loads(f) for f in json_list]
    
    data_paths = os.path.join(args.data_dir, "public.jsonl")
    with  open(data_paths, 'r', encoding="utf-8") as json_file:
        json_list = list(json_file)
        data["valid"] = [json.loads(f) for f in json_list]

    return data

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    data = load_dataset(args)

    train_dataset = SummarizeDataset(data["train"], tokenizer, args.max_source_length, args.max_target_length, 'train')
    valid_dataset = SummarizeDataset(data["valid"], tokenizer, args.max_source_length, args.max_target_length, 'valid')


    training_args = Seq2SeqTrainingArguments(
        output_dir=args.ckpt_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=2,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epoch,
        # fp16=True,
        do_train=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/test/",
    )

    # data
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=64)

    # optimizer
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epoch", type=float, default=30)

    # data loader
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
