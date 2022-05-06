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

        # inputs, targets = [], []
        # for i in range(len(self.data)):
        #     if self.data[i][text_column] is not None and self.data[i][summary_column] is not None:
        #         inputs.append(self.data[i][text_column])
        #         targets.append(self.data[i][summary_column])

        # inputs = [prefix + inp for inp in inputs]
        # model_inputs = self.tokenizer(inputs, max_length=self.max_len, padding=padding, truncation=True)

        # # Setup the tokenizer for targets
        # with self.tokenizer.as_target_tokenizer():
        #     labels = self.tokenizer(targets, max_length=self.max_len, padding=padding, truncation=True)

        # # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # # padding in the loss.
        # if padding == "max_length":
        #     labels["input_ids"] = [
        #         [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #     ]

        # model_inputs["labels"] = labels["input_ids"]

        # return model_inputs

        self.inputs = []
        for i in range(len(self.data)):
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
                if split == "test":
                    label = self.tokenizer(target, max_length=self.max_target_length, padding=padding, truncation=True, return_tensors="pt")
                else:
                    label = self.tokenizer(input, max_length=self.max_target_length, padding=padding, truncation=True)
            model_input["labels"] = label["input_ids"]
            if split == "test":
                model_input["id"] = self.data[i]["id"]
            self.inputs.append(model_input)


    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index]
