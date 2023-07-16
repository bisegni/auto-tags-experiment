from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import numpy as np
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Load the tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)  # num_labels is the number of your tags
texts = [" text1", " text2"]
tags = [["tag1"], ["tag2"]]
# Initialize MultiLabelBinarizer for tags
mlb = MultiLabelBinarizer()
mlb.fit(tags)  # Assuming `tags` is your initial set of tags

# Tokenize your initial texts
inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Convert tags to tensor format
labels = torch.tensor(mlb.transform(tags))

# Put these lines before creating your model
device = torch.device('cpu')

# Then put your model on this device
model = model.to(device)

# Make sure your inputs are on the same device
inputs = inputs.to(device)
labels = labels.to(device)

# Create a PyTorch Dataset
class TextTagDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

dataset = TextTagDataset(inputs, labels)

# Train the model with initial data
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',        
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=dataset,         
)


trainer.train()

# Now, to update the model with new data:
new_texts = ["new text1", "new text2"]
new_tags = [["new_tag1"], ["new_tag2"]]

# Update mlb with new tags
mlb.fit(np.concatenate([mlb.classes_, new_tags]))

# Tokenize your new texts
new_inputs = tokenizer(new_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Convert new_tags to tensor format
new_labels = torch.tensor(mlb.transform(new_tags))

# Create a new PyTorch Dataset
new_dataset = TextTagDataset(new_inputs, new_labels)

# Train the model with the new data
trainer = Trainer(
    model=model,                        
    args=training_args,                 
    train_dataset=new_dataset,         
)

trainer.train()

# Predict tags for a new text
new_text = ["some new text"]
new_input = tokenizer(new_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

model.eval()
with torch.no_grad():
    logits = model(**new_input).logits
    probabilities = torch.sigmoid(logits)
    predicted_tags = mlb.inverse_transform(probabilities > 0.5)  # use a threshold to convert probabilities to binary vectors
    print(predicted_tags)
