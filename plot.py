import os
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataset import SummarizeDataset
from tw_rouge import get_rouge
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)


def load_dataset(args):
    data = {}

    with  open(args.file_path, 'r', encoding="utf-8") as json_file:
        json_list = list(json_file)
        data["test"] = [json.loads(f) for f in json_list]
    return data

def main(args):
    dataset = load_dataset(args)

    rouge_1 = []
    rouge_2 = []
    rouge_L = []

    refs = {}
    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title'].strip() + '\n'
    keys =  refs.keys()
    refs = [refs[key] for key in keys]

    for i in range(args.start_ckpt, args.end_ckpt, args.period):
        ckpt_path = os.path.join(args.ckpt_dir, "checkpoint-"+str(i))
        print(ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(args.device)

        test_dataset = SummarizeDataset(dataset["test"], tokenizer, args.max_source_length, args.max_target_length, 'test')
        dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        # predict
        preds = {}
        for data in tqdm(dataloader):
            with torch.no_grad():
                predict = model.generate(
                    input_ids=data['input_ids'].squeeze().to(args.device),
                    attention_mask=data['attention_mask'].squeeze().to(args.device),
                    max_length=args.max_source_length,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )

                predict = [tokenizer.decode(p, skip_special_tokens=True) for p in predict]
                for title, id in zip(predict, data["id"]):
                    preds[id] = title.strip() + '\n'
        preds = [preds[key] for key in keys]

        # eval
        rouge = get_rouge(preds, refs)
        rouge_1.append(rouge["rouge-1"]['f'] * 100)
        rouge_2.append(rouge["rouge-2"]['f'] * 100)
        rouge_L.append(rouge["rouge-l"]['f'] * 100)
        print(rouge)


    plt.plot(
        list(range(args.start_ckpt, args.end_ckpt, args.period)),
        rouge_1,
        color="orange",
        label="rouge-1",
    )
    plt.plot(
        list(range(args.start_ckpt, args.end_ckpt, args.period)),
        rouge_2, 
        color="green", 
        label="rouge-2"
    )
    plt.plot(
        list(range(args.start_ckpt, args.end_ckpt, args.period)),
        rouge_L, 
        color="blue", 
        label="rouge-L"
    )
    plt.title("learning curve")
    plt.xlabel("steps")
    plt.ylabel("ROUGE")
    plt.legend()
    plt.savefig("ROUGE_curve.png")
            
            

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to load the model file.",
        default="./ckpt/mt5/",
    )
    parser.add_argument("--file_path", type=str, default='public.jsonl')

    # data
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=64)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)

    # ground-truth
    parser.add_argument('-r', '--reference')

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )

    parser.add_argument('-s', "--start_ckpt", type=int, default=500)
    parser.add_argument('-e', "--end_ckpt", type=int, default=40501)
    parser.add_argument('-p', "--period", type=int, default=4000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
