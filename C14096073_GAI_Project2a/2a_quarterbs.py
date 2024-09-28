import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
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
        # Convert target to class label
        class_label = min(max(tgt, -198), 297) - (-198)  # Convert to range [0, 495]
        encoding = self.tokenizer(src, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        return input_ids, torch.tensor(class_label)

# Set hyperparameters.
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
MAX_LENGTH = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 30
NUM_CLASSES = 496  # Number of classes, considering the range -198 to 297

# Load the model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = CustomModel(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, tokenizer=tokenizer, num_classes=NUM_CLASSES)

# Using CUDA if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Training on {device}.")

## Load dataset.
data = pd.read_csv(os.path.join('dataset1.csv'))

# Create virtual labels based on the sample counts of each case.
virtual_labels = []
case_lengths = [50000, 50000]
for i, length in enumerate(case_lengths):
    virtual_labels.extend([f'case{i+1}'] * length)

# Split dataset into training and validation sets with stratified sampling.
train_data, val_data = train_test_split(data, test_size=0.1, stratify=virtual_labels)

# Create custom datasets.
train_dataset = CustomDataset(train_data, tokenizer, MAX_LENGTH, NUM_CLASSES)
val_dataset = CustomDataset(val_data, tokenizer, MAX_LENGTH, NUM_CLASSES)

# Create data loaders.
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define loss function and optimizer.
criterion = torch.nn.CrossEntropyLoss()  # Using Cross Entropy Loss for classification
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Accuracy function to evaluate the model.
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).float().sum()
    accuracy = correct / len(labels)
    return accuracy.item()

# Lists to store training and validation losses and accuracies.
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_loss = float('inf')
# Training loop.
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    total_train_accuracy = 0.0
    for inputs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        accuracy = calculate_accuracy(outputs, labels)
        total_train_accuracy += accuracy
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_dataloader)}, Train Accuracy: {total_train_accuracy / len(train_dataloader)}")
    train_losses.append(train_loss/len(train_dataloader))
    train_accuracies.append(total_train_accuracy / len(train_dataloader))

    # Validation loop.(Evaluate)
    model.eval()
    val_loss = 0.0
    total_val_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            total_val_accuracy += accuracy
    
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_dataloader)}, Validation Accuracy: {total_val_accuracy / len(val_dataloader)}")
    val_losses.append(val_loss/len(val_dataloader))
    val_accuracies.append(total_val_accuracy / len(val_dataloader))

    # Save the model if validation loss decreases.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model, "best_model_quarterbs.pth")

# Plot both the training and validation loss curves on the same figure.
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig("training_validation_curves_quarterbs.png")
plt.show()
