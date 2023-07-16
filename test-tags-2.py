import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import random

# Step 2: Prepare your dataset
class CustomDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, max_length, tag_list):
        self.tokenizer = tokenizer
        self.texts = texts
        self.tags = tags
        self.max_length = max_length
        self.tag_list = tag_list

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        tags = self.tags[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        tags_tensor = self.get_tags_tensor(tags)

        return input_ids, attention_mask, tags_tensor

    def get_tags_tensor(self, tags):
        tags_tensor = torch.zeros(len(self.tag_list))
        for tag in tags:
            if tag in self.tag_list:
                tag_index = self.tag_list.index(tag)
                tags_tensor[tag_index] = 1

        return tags_tensor

# Generate example text and tags for training
train_texts = [
    "I love playing soccer",
    "Today is a beautiful day",
    "I enjoy reading books",
    "The stock market is crashing",
    "Let's go hiking this weekend",
    "New recipe for delicious pasta"
]

train_tags = [
    ["sports", "hobbies"],
    ["weather"],
    ["hobbies"],
    ["finance"],
    ["hobbies"],
    ["food", "recipes"]
]

# Generate example text and tags for test
test_texts = [
    "I can't wait for the basketball game",
    "The sun is shining brightly",
    "Gardening is my favorite hobby",
    "The economy is booming",
    "Planning a road trip with friends",
    "Try this amazing cake recipe"
]

test_tags = [
    ["sports", "hobbies"],
    ["weather"],
    ["hobbies"],
    ["finance"],
    ["hobbies"],
    ["food", "recipes"]
]

# Step 3: Preprocess your data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128

# Create a list of all unique tags
all_tags = list(set(tag for tags in train_tags + test_tags for tag in tags))

# Step 4: Create data loaders for training and test data
train_dataset = CustomDataset(train_texts, train_tags, tokenizer, max_length, all_tags)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = CustomDataset(test_texts, test_tags, tokenizer, max_length, all_tags)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 5: Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(all_tags))

# Step 6: Fine-tune the BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for batch in train_loader:
        input_ids, attention_mask, tags_tensor = batch
        input_ids, attention_mask, tags_tensor = input_ids.to(device), attention_mask.to(device), tags_tensor.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        predicted = (torch.sigmoid(logits) > 0.5).int()
        train_correct += (predicted == tags_tensor).sum().item()

        loss = loss_fn(logits, tags_tensor.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_accuracy = train_correct / len(train_dataset)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

# Step 7: Evaluation
model.eval()
test_correct = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, tags_tensor = batch
        input_ids, attention_mask, tags_tensor = input_ids.to(device), attention_mask.to(device), tags_tensor.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        predicted = (torch.sigmoid(logits) > 0.5).int()
        test_correct += (predicted == tags_tensor).sum().item()

test_accuracy = test_correct / len(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")
torch.save(model.state_dict(), 'path_to_trained_model.pth')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(all_tags))
model.load_state_dict(torch.load('path_to_trained_model.pth'))
model.eval()
input_text = "i read a book"
encoded_input = tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,
    max_length=max_length,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted = (torch.sigmoid(logits) > 0.5).int()
predicted_tags = [all_tags[i] for i, pred in enumerate(predicted[0]) if pred == 1]
print(predicted_tags)