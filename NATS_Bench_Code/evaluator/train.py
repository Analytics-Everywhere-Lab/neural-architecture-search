# Trains just the encoder

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import csv

import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils import Encoder, MakeDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--search_space', type=str, default='tss', help='Either tss or sss')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, or ImageNet16-120')
parser.add_argument('--metric', type=str, default='Latency', help='Memory or Latency (case sensitive)')
args = parser.parse_args()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained('t5-large')
scaler = MinMaxScaler()

df = pd.read_csv(f'../{args.search_space}_{args.dataset}.csv')

df_temp, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(df_temp, test_size=0.25, random_state=42)

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

# Determines how the data is structured. Either (acc, lat) or (acc, mem)
if args.metric == 'Latency':
    train_dataset = MakeDataset(train, tokenizer, scaler, latency=True, memory=False)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = MakeDataset(val, tokenizer, scaler, latency=True, memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

if args.metric == 'Memory':
    train_dataset = MakeDataset(train, tokenizer, scaler, latency=False, memory=True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = MakeDataset(val, tokenizer, scaler, latency=False, memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)



model = Encoder()
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()

history= []

for epoch in range(20):
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        input_ids = batch['input_ids']
        labels = batch['metrics']

        input_ids, labels = input_ids.to(device), labels.to(device)

        encoder_outputs, prediction_outputs = model(input_ids=input_ids)

        regression_loss = nn.MSELoss()(prediction_outputs, labels)
        regression_loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Batch: {i}, Loss: {regression_loss.item()}')

        if i%200 == 0: 
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _, val_batch in enumerate(val_dataloader):
                    val_input_ids = val_batch['input_ids']
                    val_labels = val_batch['metrics']

                    val_input_ids, val_labels = val_input_ids.to(device), val_labels.to(device)

                    val_outputs, val_prediction = model(input_ids=val_input_ids)

                    val_loss += nn.MSELoss()(val_prediction, val_labels)

            print(f'Epoch: {epoch}, Batch: {i}, Loss: {regression_loss.item()}, Val_Loss: {val_loss.item()}')

            history.append([epoch, i, regression_loss.item(), val_loss.item()])
            # with open('sss_mem_train_history.csv', mode='w', newline='') as file: # Memory training 
            with open(f'{args.search_space}_history/{args.dataset}_{args.metric}_training.csv', mode='w', newline='') as file: # Latency training
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Batch', 'Train_loss', 'Val_loss'])
                writer.writerows(history)
        model.train()
            
torch.save(model.state_dict(), f'saved_models/{args.search_space}_{args.dataset}_{args.metric}.pth')

# Evaluate on the test set
save_data = []
model.eval()
for i in range(len(test)):
    test_sample = test.iloc[i]
    test_str = test_sample['Architectures']
    test_metrics = test_sample[['Accuracy', f'{args.metric}']].values
    
    tokenized_test_str = tokenizer(test_str, max_length=None, truncation=True, padding='max_length', return_tensors='pt').input_ids.to(device)
    _, prediction_outputs = model(tokenized_test_str)
    prediction_outputs = prediction_outputs.detach().cpu().numpy()
    prediction_outputs = scaler.inverse_transform(prediction_outputs)
    print(i, test_metrics, prediction_outputs)

    
    save_data.append([i, test_str, float(test_sample['Accuracy']), float(test_sample[f'{args.metric}']),  prediction_outputs[:, 0], prediction_outputs[:, 1]]) 
    
    with open(f'{args.search_space}_history/{args.dataset}_{args.metric}_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(["Batch", "Original Text", "True Accuracy", "True Memory", "Predicted Accuracy", "Predicted Memory"])  
        writer.writerow(['Batch', 'Original Text', 'True Accuracy', f'True {args.metric}', 'Predicted Accuracy', f'Predicted {args.metric}'])
        writer.writerows(save_data)  # Write data

