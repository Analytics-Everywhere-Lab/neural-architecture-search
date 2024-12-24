# Trains both the encoder and decoder at once. To be worked on later

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv

import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils import EncoderDecoder, MakeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained('t5-small')
scaler = MinMaxScaler()

df = pd.read_csv('trainset.csv')
# arch = df['Architecture']
# metric = df[['Accuracy', 'memory']]

text_lengths = [len(tokenizer.tokenize(text)) for text in df['Architecture']]
max_length = max(text_lengths)

split_index = int(0.8 * len(df))
train = df[:split_index]
test = df[split_index:]

train_dataset = MakeDataset(train, tokenizer, scaler)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

test_dataset = MakeDataset(test, tokenizer, scaler)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = EncoderDecoder()
# model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()

# Encoder-only training
# for epoch in range(5):
#     for batch in dataloader:
#         optimizer.zero_grad()

#         input_ids = batch['input_ids']
#         labels = batch['metrics']
#         # decoder_input_ids = batch['decoder_input_ids']

#         value_outputs = model(input_ids=input_ids)
#         regression_loss = nn.MSELoss()(value_outputs, labels)

#         # decoder_labels = batch['decoder_input_ids']
#         # reconstruction_loss = nn.CrossEntropyLoss()(decoder_outputs, decoder_labels)

#         total_loss = regression_loss 
#         total_loss.backward()
#         optimizer.step()

#         print(f'Epoch: {epoch}, Loss: {total_loss.item()}')

save_data = []
# Encoder-Decoder training
for epoch in range(20):
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        input_ids = batch['input_ids']
        # attention_mask = batch['attention_mask']
        labels = batch['metrics']
        decoder_input_ids = batch['decoder_input_ids']

        input_ids, labels, decoder_input_ids = input_ids.to(device), labels.to(device), decoder_input_ids.to(device)

        encoder_outputs, prediction_outputs, decoder_outputs, decoder_logits = model(
            input_ids=input_ids, 
            encoder_outputs=None,
            decoder_input_ids=decoder_input_ids
        )

        regression_loss = nn.MSELoss()(prediction_outputs, labels)

        decoder_labels = batch['decoder_input_ids'].to(device)
        # print(decoder_labels.shape(), decoder_outputs.shape())
        reconstruction_loss = nn.CrossEntropyLoss()(decoder_logits.view(-1, decoder_logits.size(-1)), decoder_labels.view(-1))

        total_loss = regression_loss + reconstruction_loss
        total_loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch}, Batch: {i}, Loss: {total_loss.item()}')
        # print(decoder_input_ids.shape)
        if i%100 == 0:
            # PRINTING GENERATED TEXT AT INTERVALS#
            # test_str = f'{tokenizer.decode(input_ids[0], skip_special_tokens=True)}' 
            test_str = test['Architecture'].sample(n=1).iloc[0]
            
            print(f'Original Text: \n{test_str}\n')
            first_substring = test_str.split('|')[1].strip()
            first_string_input_ids = tokenizer(first_substring, add_special_tokens=False, return_tensors="pt").input_ids # BOS token
            
            for _ in range(len(test_str)):
                first_string_input_ids = first_string_input_ids.to(device)
                encoder_outputs, _, _, decoder_logits = model(input_ids, encoder_outputs=None, decoder_input_ids=first_string_input_ids)
                next_decoder_inputs = torch.argmax(decoder_logits[:, -1], axis=-1).unsqueeze(1).to(device)
                first_string_input_ids = torch.cat([decoder_input_ids, next_decoder_inputs], axis=-1)      
                # print(f'{tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)}')

                if next_decoder_inputs[0, 0].item() == tokenizer("</s>", add_special_tokens=False, return_tensors="pt").input_ids: # EOS
                    break
            
            # print(input_ids.shape[0])
            # for i in range(input_ids.shape[0]):
            gen_text = tokenizer.decode(first_string_input_ids[0], skip_special_tokens=True)
            print(f'Generated Text: \n{gen_text}\n')

            save_data.append([i, test_str, gen_text, regression_loss.item(), reconstruction_loss.item(), total_loss.item()])

save_path = 'transformer/enc_dec_train_history.csv'

with open(save_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Batch", "Original Text", "Generated Text", "Regression Loss", "Reconstruction Loss", "Total Loss"])  # Write header
    writer.writerows(save_data)  # Write data

torch.save(model.state_dict(), 'T5SavedEncoderDecoder.pth')