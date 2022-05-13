import os
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataset import SummarizeDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
)

def load_dataset(args):
    data = {}
    with  open(args.file_path, 'r', encoding="utf-8") as json_file:
        json_list = list(json_file)
        data["test"] = [json.loads(f) for f in json_list]

    return data

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt_dir).to(args.device)

    data = load_dataset(args)

    test_dataset = SummarizeDataset(data["test"], tokenizer, args.max_source_length, args.max_target_length, 'test')

    dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # predict
    preds = []
    for data in tqdm(dataloader):
        with torch.no_grad():
            predict = model.generate(
                input_ids=data['input_ids'].squeeze().to(args.device),
                attention_mask=data['attention_mask'].squeeze().to(args.device),
                max_length=args.max_source_length,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

            predict = [tokenizer.decode(p, skip_special_tokens=True) for p in predict]
            preds += [{'title': title, 'id': id} for title, id in zip(predict, data["id"])]
    
    # output
    with open(args.output_path, "w") as json_file:
        for pred in preds:
            print(json.dumps(pred, ensure_ascii=False), file=json_file)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/mt5/",
    )
    parser.add_argument("--file_path", type=str, default='public.jsonl')
    parser.add_argument("--output_path", type=str, default='submission.jsonl')

    # data
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=64)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)

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
