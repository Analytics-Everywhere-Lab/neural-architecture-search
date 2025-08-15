# Trains just the encoder

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import csv

import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding

from utils import Encoder, MakeDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--search_space', type=str, default='nasbench201', help='Either tss or sss')
parser.add_argument('--device', type=str, default='edgegpu', help='edgegpu, edgetpu, eyeriss, fpga, pixel3, raspi4')
args = parser.parse_args()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained('t5-base')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")  # Max add
scaler = MinMaxScaler()

df = pd.read_csv(f'profiled_nets/{args.search_space}_{args.device}.csv')

df_temp, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(df_temp, test_size=0.25, random_state=42)

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

train_dataset = MakeDataset(train, tokenizer, scaler, accuracy=False) 
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)  # Max change

val_dataset = MakeDataset(val, tokenizer, scaler, accuracy=False)
# val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)  # Max change

model = Encoder(num_outputs=1)
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
            
            with open(f'history/{args.device}_train_history.csv', mode='w', newline='') as file: # Latency training
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Batch', 'Train_loss', 'Val_loss'])
                writer.writerows(history)
        model.train()

torch.save(model.state_dict(), f'saved_models/{args.device}.pth')

save_data = []
model.eval()
for i in range(len(test)):
    test_sample = test.iloc[i]
    test_str = test_sample['Architectures']
    test_metrics = test_sample['Latency'] 
    
    tokenized_test_str = tokenizer(test_str, max_length=None, truncation=True, padding='max_length', return_tensors='pt').input_ids.to(device)
    _, prediction_outputs = model(tokenized_test_str)
    prediction_outputs = prediction_outputs.detach().cpu().numpy()
    prediction_outputs = scaler.inverse_transform(prediction_outputs) # Removed for FBNet
    print(i, test_metrics, prediction_outputs)

    save_data.append([i, test_str, float(test_sample['Latency']),  prediction_outputs[:, 0]]) 

    save_path = f'history/{args.device}test_results.csv'

    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Batch", "Original Text", "True Latency", "Predicted Latency"])
        writer.writerows(save_data)  