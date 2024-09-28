import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# Define your custom model.
class CustomModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tokenizer, num_classes):
        super(CustomModel, self).__init__()
        self.embedding_layer = torch.nn.Embedding(num_embeddings=tokenizer.vocab_size,
                                                  embedding_dim=embedding_dim)
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)  # Output layer now has num_classes neurons

    def forward(self, x):
        embedded = self.embedding_layer(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output[:, -1, :])
        return output.squeeze(1)

# Define your custom dataset.
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, num_classes):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = str(self.data.iloc[idx, 0])
        tgt = int(self.data.iloc[idx, 1])  # Convert target to integer
        class_label = min(max(tgt, -198), 297) - (-198)  # Convert to range [0, 495]
        encoding = self.tokenizer(src, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        return input_ids, torch.tensor(class_label)


# Accuracy function to evaluate the model.
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).float().sum()
    accuracy = correct / len(labels)
    return accuracy.item()



    

# Set hyperparameters.
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
MAX_LENGTH = 30
NUM_CLASSES = 496

# Load the model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = CustomModel(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, tokenizer=tokenizer, num_classes=NUM_CLASSES)

# Using CUDA if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Training on {device}.")

# Load the saved model.
model = torch.load("best_model_BASE_DATASET2.pth").to(device)

# Define loss function and optimizer.
criterion = torch.nn.CrossEntropyLoss()

# Load test datasets and evaluate the model on each.
for i in range(4, 7):
    test_data = pd.read_csv(os.path.join(f'test{i}.csv'))
    test_dataset = CustomDataset(test_data, tokenizer, MAX_LENGTH, NUM_CLASSES)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    test_loss = 0.0
    total_test_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc=f'Testing Dataset {i}'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            total_test_accuracy += accuracy

    print(f"Test{i} Loss: {test_loss/len(test_dataloader)}, Test{i} Accuracy: {total_test_accuracy / len(test_dataloader)}")
