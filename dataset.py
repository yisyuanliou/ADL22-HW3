from tqdm import tqdm
from torch.utils.data import Dataset

class SummarizeDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length, max_target_length, split):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.split = split

        self.preprocess_function(split)

    # preprocessing
    def preprocess_function(self, split, padding="max_length"):
        # remove pairs where at least one record is None
        text_column = "maintext"
        summary_column = "title"
        prefix = "summarize: "

        self.inputs = []
        for i in tqdm(range(len(self.data))):
            if self.data[i][text_column] is not None and self.data[i][summary_column] is not None:
                input = self.data[i][text_column]
                target = self.data[i][summary_column]
            input = prefix + input
            if split == "test":
                model_input = self.tokenizer(input, max_length=self.max_source_length, padding=padding, truncation=True, return_tensors="pt")
            else:
                model_input = self.tokenizer(input, max_length=self.max_source_length, padding=padding, truncation=True)
            
            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                if split != "test":
                    label = self.tokenizer(target, max_length=self.max_target_length, padding=padding, truncation=True)
                    model_input["labels"] = label["input_ids"]

            if split == "test":
                model_input["id"] = self.data[i]["id"]
            self.inputs.append(model_input)


    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index]
