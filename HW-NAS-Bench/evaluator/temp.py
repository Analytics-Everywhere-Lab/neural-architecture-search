# Supplements train.py to run concurrently

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained('t5-base')
scaler = MinMaxScaler()

df = pd.read_csv('profiled_nets/fbnet_eyeriss.csv') # Replace as necessary

df_temp, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(df_temp, test_size=0.25, random_state=42)

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

train_dataset = MakeDataset(train, tokenizer, scaler, accuracy=False) # Set score to True for NASWOT score, and latency true for latency metric
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = MakeDataset(val, tokenizer, scaler, accuracy=False)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

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
            val_loss = None
            with torch.no_grad():
                for _, val_batch in enumerate(val_dataloader):
                    val_input_ids = val_batch['input_ids']
                    val_labels = val_batch['metrics']

                    val_input_ids, val_labels = val_input_ids.to(device), val_labels.to(device)
                    
                    val_outputs, val_prediction = model(input_ids=val_input_ids)

                    val_loss = nn.MSELoss()(val_prediction, val_labels)
            print(f'Epoch: {epoch}, Batch: {i}, Loss: {regression_loss.item()}, Val_Loss: {val_loss.item()}')

            history.append([epoch, i, regression_loss.item(), val_loss.item()])
            # with open('sss_mem_train_history.csv', mode='w', newline='') as file: # Memory training 
            with open('history/fbnet/eyeriss_train_history.csv', mode='w', newline='') as file: # Latency training
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Batch', 'Train_loss', 'Val_loss'])
                writer.writerows(history)
            
# torch.save(model.state_dict(), 'saved_models/sss_mem_model.pth')
torch.save(model.state_dict(), 'saved_models/fbnet/eyeriss_model.pth')

save_data = []
for i in range(len(test)):
    test_sample = test.iloc[i]
    test_str = test_sample['Architectures']
    # test_metrics = test_sample[['Accuracy', 'Latency']].values  # NASBench201
    test_metrics = test_sample['Latency']  # FBNet
    tokenized_test_str = tokenizer(test_str, max_length=None, truncation=True, padding='max_length', return_tensors='pt').input_ids.to(device)
    _, prediction_outputs = model(tokenized_test_str)
    prediction_outputs = prediction_outputs.detach().cpu().numpy()
    prediction_outputs = scaler.inverse_transform(prediction_outputs)
    print(i, test_metrics, prediction_outputs)

    # save_data.append([i, test_str, float(test_sample['Accuracy']), float(test_sample['Latency']),  prediction_outputs[:, 0], prediction_outputs[:, 1]]) # NASBench 201
    save_data.append([i, test_str, float(test_sample['Latency']),  prediction_outputs[:, 0]]) #FBNet

    # save_path = 'sss_mem_test_results.csv'
    save_path = 'history/fbnet/eyeriss_test_results.csv'

    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(["Batch", "Original Text", "True Accuracy", "True Memory", "Predicted Accuracy", "Predicted Memory"])  
        writer.writerow(["Batch", "Original Text", "True Latency", "Predicted Latency"])
        writer.writerows(save_data)  # Write data

