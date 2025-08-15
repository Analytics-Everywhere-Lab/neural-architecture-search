from transformers import T5Model
from torch import nn
import torch
from torch.utils.data import Dataset

# class EncoderDecoder(nn.Module):
#     def __init__(self, model_name='t5-large'):
#         super(EncoderDecoder, self).__init__()
#         self.model = T5Model.from_pretrained(model_name)
#         self.encoder = self.model.encoder
#         self.decoder = self.model.decoder

#         self.regression_head = nn.Linear(self.encoder.config.d_model, 2)

#     def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
#         # Take the mean across the sequence dim to reduce to a single vector per batch
#         # The output is a sequence of hidden states for each token in the input sequence
#         # shape is [batch_size, seq_length, hidden_dim]
#         # dim collapses the sequence of vectors into a single vector per batch with shape [batch_size, hidden_dim]
#         encoder_outputs = self.model.encoder(input_ids, return_dict=True).last_hidden_state.mean(dim=1)
#         prediction_outputs = self.regression_head(encoder_outputs)

#         return prediction_outputs
    

class MakeDataset(Dataset):
    def __init__(self, df, tokenizer, scaler, latency=False, memory=False):
        self.df = df
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.archs = self.df['Architectures']
 
        if latency == True:
            self.metrics = list(zip(self.df['Accuracy'], self.df['Latency']))
        if memory == True:
            self.metrics = list(zip(self.df['Accuracy'], self.df['Memory']))
        self.scaled_metrics = scaler.fit_transform(self.metrics)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        arch = self.archs[idx]
        acc, metric = self.scaled_metrics[idx]

        inputs = self.tokenizer(arch, max_length=None, truncation=True, padding='max_length', return_tensors='pt').input_ids
        targets = self.tokenizer(arch, max_length=None, truncation=True, padding='max_length', return_tensors='pt').input_ids
    
        metric_tensor = torch.tensor([acc, metric], dtype=torch.float)

        # target_str = f'{acc} | {mem}'
        # targets = self.tokenizer(target_str, max_length=10, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': inputs.squeeze(), # .squeeze() removes all dimensions of size 1
            # 'attention_mask': inputs['attention_mask'].squeeze(),
            'metrics': metric_tensor,
            'decoder_input_ids': targets.squeeze()
        }


class EncoderDecoder(nn.Module):
    def __init__(self, model_name='t5-large'):
        super(EncoderDecoder, self).__init__()
        self.model = T5Model.from_pretrained(model_name)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.embeddings = self.model.get_input_embeddings()
        self.regression_head = nn.Linear(self.encoder.config.d_model, 2)

    def forward(self, input_ids, encoder_outputs, decoder_input_ids):
        if encoder_outputs is None: # Else condition is inference mode
            encoder_outputs = self.model.encoder(input_ids, return_dict=True)
            # encoder_hidden_states = encoder_outputs
        # print(type(encoder_outputs))
        # decoder_input_ids.to("cuda")
        prediction_outputs = self.regression_head(encoder_outputs.last_hidden_state.mean(dim=1))
        # if isinstance(encoder_outputs, torch.Tensor): # In inference, the last
        #     print('Generating')
        #     decoder_outputs = self.decoder(decoder_input_ids, encoder_hidden_states=encoder_outputs).last_hidden_state
        
        
        decoder_outputs = self.decoder(decoder_input_ids, encoder_hidden_states=encoder_outputs.last_hidden_state).last_hidden_state
        decoder_logits = nn.functional.linear(decoder_outputs, self.embeddings.weight)
        return encoder_outputs, prediction_outputs, decoder_outputs, decoder_logits
    
class Encoder(nn.Module):
    def __init__(self, model_name='t5-large', num_outputs=2): # Vary the number of outputs according to how many target metrics you want
        super(Encoder, self).__init__()
        self.model = T5Model.from_pretrained(model_name)
        self.encoder = self.model.encoder
        self.regression_head = nn.Linear(self.encoder.config.d_model, num_outputs)

    def forward(self, input_ids):
        encoder_outputs = self.encoder(input_ids, return_dict=True)
        prediction_outputs = self.regression_head(encoder_outputs.last_hidden_state.mean(dim=1))
        return encoder_outputs, prediction_outputs