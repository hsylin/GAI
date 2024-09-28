#reference https://github.com/minmie/gpt2-summarization
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, default_data_collator, \
    DataCollatorWithPadding, get_linear_schedule_with_warmup,GPT2LMHeadModel
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import jieba
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertTokenizer
from ignite.metrics import Rouge
def collator_fn(batch):
    batch_data = [each.values() for each in batch]
    input_ids, labels= zip(*batch_data)
    input_ids = [torch.tensor(i, dtype=torch.long) for i in input_ids]
    labels = [torch.tensor(i, dtype=torch.long) for i in labels]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=2).to(device)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)# Ignoring loss for padding tokens
    return {
            "input_ids": input_ids,
            "labels": labels,
        }

        
def remove_ignore_tokens_from_list(tensor_list, ignore_token=-100):
    filtered_tensor_list = [tensor[tensor != ignore_token] for tensor in tensor_list]
    return filtered_tensor_list

def find_first_index_in_nested_lists(nested_lists):
    result = [next((i for i, num in enumerate(lst) if num != -100), None) for lst in nested_lists]
    return result


def evaluate(model):
    pbar = tqdm(valid_dataloader)
    pbar.set_description(f"Evaluating")
    rouge.reset()
    for data in pbar:
        inputs = data["input_ids"].to(device)
        targets = data["labels"].to(device)
        ignore_tokens_count =find_first_index_in_nested_lists(targets)
        output = model.generate(input_ids=inputs, max_length=512, pad_token_id=50256)
        output_no_ignore = [output[i, ignore_tokens_count[i]:] for i in range(output.shape[0])]
        output_texts = [tokenizer.batch_decode(output_no_ignore, skip_special_tokens=True)]
        targets = remove_ignore_tokens_from_list(targets, ignore_token=-100)
        targets = [tokenizer.batch_decode(targets, skip_special_tokens=True)]
        for i in range(len(output_texts)):
            sentences = [s.replace('。', ' 。').split() for s in output_texts[i]]
            ground_truths = [t.replace('。', ' 。').split() for t in targets[i]]
            for s in sentences:
                rouge.update(([s], [ground_truths]))           
    return rouge.compute()

#process train data
def process(examples):

    sample_input_ids = [tokenizer.cls_token_id]+tokenizer.encode(examples["text"], add_special_tokens=False)+[tokenizer.cls_token_id]
    label_input_ids = [tokenizer.sep_token_id]+tokenizer.encode( examples["summary"], add_special_tokens=False)+[tokenizer.sep_token_id]
    input_ids = sample_input_ids + label_input_ids
    labels = [-100] * len(sample_input_ids) + label_input_ids
    assert len(input_ids) == len(labels)
    return {
    "input_ids": input_ids,
    "labels": labels,
}

#process data -->evaluate
def process2(examples):

    sample_input_ids = [tokenizer.cls_token_id]+tokenizer.encode(examples["text"], add_special_tokens=False)+[tokenizer.cls_token_id]
    label_input_ids = [tokenizer.sep_token_id]+tokenizer.encode( examples["summary"], add_special_tokens=False)+[tokenizer.sep_token_id]
    input_ids = sample_input_ids
    labels = [-100] * len(sample_input_ids) + label_input_ids
    return {
    "input_ids": input_ids,
    "labels": labels,
}


# Set hyperparameters.
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 40

# Initialize tokenizer and model
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall").cuda() if torch.cuda.is_available() else GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
rouge = Rouge(variants=[1,2,"L"], multiref="best")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# Check and print the device being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset from JSON  file 
data = load_dataset("hugcyp/LCSTS")
train_data = data["train"]
valid_data=  data["validation"]
train_data = train_data.select(range(3200))
valid_data = valid_data.select(range(320))

# Create dataset and data loader
train_dataset = train_data.map(process,remove_columns=train_data.column_names)
train_dataloader = DataLoader( train_dataset, collate_fn=collator_fn, batch_size=BATCH_SIZE,shuffle=True)
valid_dataset = valid_data.map(process2, remove_columns=valid_data.column_names)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collator_fn, batch_size=BATCH_SIZE, shuffle=False)

# Initialize optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
      
# Training loop      
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    with tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch') as train_pbar:
        for batch in train_pbar:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()  
            total_loss += loss.item()
            train_pbar.set_postfix({'Loss': total_loss / (len(train_pbar))})
    # Save model at the end of each epoch
    folder_name = "model_gpt2"   
    os.makedirs(folder_name, exist_ok=True)
    torch.save(model, f"{folder_name}/model_gpt2_epoch_{epoch}.pt")
    
    # Evaluate model after each epoch
    print(f"Rouge score on epoch {epoch}:", evaluate(model))

    
