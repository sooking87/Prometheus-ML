import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Tuple, Union, Optional

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import get_tokenizer

import string ,re
from unicodedata import normalize

print(th.cuda.get_device_name(), th.cuda.get_device_capability())
device = th.device("cuda" if th.cuda.is_available() else "cpu")
# device = th.device('cpu')
print(device)


en_tokenizer = get_tokenizer('spacy', language='en')
fr_tokenizer = get_tokenizer('spacy', language='fr')

en_vocab = pd.read_pickle("data/en_vocab.pkl")
fr_vocab = pd.read_pickle("data/fr_vocab.pkl")

print(type(en_vocab))
print(len(en_vocab))
print(en_vocab['<pad>'])

train_data = pd.read_pickle("data/train_data.pkl")
valid_data = pd.read_pickle("data/valid_data.pkl")
test_data = pd.read_pickle("data/test_data.pkl")

print(len(train_data), len(valid_data), len(test_data))
print(type(train_data))

#### Cleaner ####
def clean_lines(lines: List[str]) -> List[str]:
    if isinstance(lines, list):
        return [clean_lines(line) for line in lines]

    is_question = lines.endswith('?')
    remove_punctuation = str.maketrans('', '', string.punctuation)
    lines = normalize('NFD', lines).encode('ascii', 'ignore')
    lines = lines.decode('UTF-8')
    lines = lines.lower()
    lines = lines.translate(remove_punctuation)
    lines = re.sub(rf'[^{re.escape(string.printable)}]', '', lines)

    lines = [word for word in lines.split() if word.isalpha()]
    if is_question:
        lines.append('?')
    return ' '.join(lines)

#### Data Loader ####
batch_size = 132
PAD_IDX = en_vocab['<pad>']
SOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

def generate_batch(data_batch):
    en_batch, fr_batch = [], []
    for (en_item, fr_item) in data_batch:
        en_batch.append(th.cat([th.tensor([SOS_IDX]), en_item, th.tensor([EOS_IDX])], dim=0))
        fr_batch.append(th.cat([th.tensor([SOS_IDX]), fr_item, th.tensor([EOS_IDX])], dim=0))
    en_batch = pad_sequence(en_batch, batch_first=True, padding_value=PAD_IDX)
    fr_batch = pad_sequence(fr_batch, batch_first=True, padding_value=PAD_IDX)
    return en_batch, fr_batch


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
val_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)


#### Model ####
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, encoder_hid_dim: int, decoder_hid_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, encoder_hid_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(encoder_hid_dim * 2, decoder_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: th.Tensor):
        """
        src: [seq_len, batch_size]
        output: [seq_len, batch_size, 2 * encoder_hid_dim], [batch_size, decoder_hid_dim]
        """
        embedded = self.dropout(self.embedding(src))  # [bs, seq_len, emb_dim]
        # outputs: [seq_len, batch_size, encoder_hid_dim * 2], hidden: [2, batch_size, encoder_hid_dim]
        outputs, hidden = self.rnn(embedded)  # [bs, seq_len, 2 * enc_hid], [2, bs, enc_hid]
        # fc input: concat last forward and backward hidden state
        hidden = th.tanh(self.fc(th.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=-1)))  # [bs, dec_hid]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, encoder_hid_dim: int, decoder_hid_dim: int, attn_dim: int):
        super().__init__()
        self.attn_input_dim = encoder_hid_dim * 2 + decoder_hid_dim
        self.attn = nn.Linear(self.attn_input_dim, attn_dim)

    def forward(self, decoder_hidden: th.Tensor, encoder_outputs: th.Tensor):
        """
        decoder_hidden: [batch_size, decoder_hid_dim]
        encoder_outputs: [batch_size, seq_len, 2 * encoder_hid_dim]
        output: [batch_size, seq_len]

        this is a concat attention
        """
        src_len = encoder_outputs.shape[1]  # 1 for batch_first else 0
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1) # [bs, seq_len, dec_hid]
        assert repeated_decoder_hidden.shape == th.Size([encoder_outputs.shape[0], src_len, decoder_hidden.shape[-1]])
        energy = self.attn(th.cat((repeated_decoder_hidden, encoder_outputs), dim=-1))  # [bs, seq_len, attn_dim]
        attention = energy.tanh().sum(dim=-1)   # [bs, seq_len]
        return F.softmax(attention, dim=-1)


class SimpleDecoder(nn.Module):
    def __init__(self, output_dim:int, emb_dim:int, encoder_hid_dim:int, decoder_hid_dim:int, dropout:float, attention:Attention):
        super().__init__()
        self.attention = attention
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + encoder_hid_dim * 2, decoder_hid_dim, batch_first=True)
        self.fc = nn.Linear(attention.attn_input_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_outputs(self, decoder_hidden: th.Tensor, encoder_outputs: th.Tensor):
        """
        decoder_hidden: [batch_size, decoder_hid_dim]
        encoder_outputs: [batch_size, seq_len, 2 * encoder_hid_dim]
        output: [batch_size, 1, 2 * encoder_hid_dim]
        """
        a = self.attention(decoder_hidden, encoder_outputs)  # [bs, seq_len]
        a = a.unsqueeze(1)
        weighted_encoder_outputs = th.bmm(a, encoder_outputs)   # [bs, 1, 2 * enc_hid]
        return weighted_encoder_outputs

    def forward(self,
                decoder_inputs:th.Tensor,
                decoder_hidden:th.Tensor,
                encoder_outputs:th.Tensor):
        """
        decoder_inputs: [batch_size,]
        hidden: [batch_size, decoder_hid_dim]
        encoder_outputs: [batch_size, seq_len, encoder_hid_dim * 2]
        output: [batch_size, output_dim]
        """
        decoder_inputs = decoder_inputs.unsqueeze(1)  # [bs, 1]
        embedded = self.dropout(self.embedding(decoder_inputs)) # [bs, 1, emb_dim]
        # output: [batch_size, 1, encoder_hid_dim], hidden: [1, batch_size, decoder_hid_dim]
        w_encoder_outputs = self._weighted_encoder_outputs(decoder_hidden, encoder_outputs) # [bs, 1, 2 * enc_hid]
        rnn_input = th.cat((embedded, w_encoder_outputs), dim=-1)   # [bs, 1, emb_dim + 2 * enc_hid]
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))   # [bs, 1, dec_hid], [1, bs, dec_hid]
        embedded = embedded.squeeze(1)  # [bs, emb_dim]
        output = output.squeeze(1)  # [bs, dec_hid]
        w_encoder_outputs = w_encoder_outputs.squeeze(1)    # [bs, 2 * enc_hid]
        output = self.fc(th.cat((output, w_encoder_outputs, embedded), dim=-1)) # [bs, output_dim]
        # output: [batch_size, output_dim]
        return output, decoder_hidden.squeeze(0)    # [bs, output_dim], [bs, dec_hid]


class Seq2Seq(nn.Module):
    def __init__(self, encoder:nn.Module, decoder:nn.Module, device:th.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src:th.Tensor, trg:th.Tensor, teacher_forcing_ratio:float=0.5):
        """
        src: [seq_len, batch_size]
        trg: [seq_len, batch_size]
        output: [seq_len, batch_size, output_dim]
        """
        batch_size = src.shape[0]
        max_len = trg.shape[1]
        output_dim = self.decoder.output_dim
        outputs = th.zeros(batch_size, max_len, output_dim).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        decoder_input = trg[:, 0]    # start with <sos>
        for t in range(1, max_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            do_teacher_force = np.random.random() < teacher_forcing_ratio
            # select one of the ground truth or the predicted word (auto-regressive)
            decoder_input = trg[:, t] if do_teacher_force else output.argmax(-1)
        return outputs

    def translate(self, src: th.Tensor, max_len: int = 100):
        with th.no_grad():
            outputs = []
            encoder_outputs, hidden = self.encoder(src)
            decoder_input = th.tensor([SOS_IDX] * 1).to(self.device)
            for t in range(1, max_len):
                output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
                outputs.append(output.argmax(-1).item())
                decoder_input = output.argmax(-1)
            return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


input_dim = len(fr_vocab)
output_dim = len(en_vocab)
enc_emb_dim = 128
dec_emb_dim = 128
enc_hid_dim = 256
dec_hid_dim = 256
attn_dim = 32
enc_dropout = 0.5
dec_dropout = 0.5

attention = Attention(enc_hid_dim, dec_hid_dim, attn_dim)
encoder = SimpleEncoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
decoder = SimpleDecoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attention)
model = Seq2Seq(encoder, decoder, device).to(device)

model.apply(init_weights)
print(model)

#### Training ####
optimizer = th.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<pad>'])

import gc

to_lang = 'en'

valid_text = "Il vaut mieux voir une fois que d'entendre mille fois."
valid_text = fr_tokenizer(clean_lines(valid_text))
valid_text = [fr_vocab['<bos>']] + [fr_vocab[token] for token in valid_text] + [fr_vocab['<eos>']]
valid_text = th.tensor(valid_text).unsqueeze(0).to(device)
valid_label = clean_lines("It is better to see once than to hear a thousand times.")

def train(model: nn.Module,
          dataloader: DataLoader,
          optimizer: th.optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc='Training', leave=False, total=len(dataloader))
    for i, (src, trg) in enumerate(progress_bar):
        if to_lang == 'en':
            src, trg = trg, src
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        if i % 5 == 0:
            gc.collect()
        th.cuda.empty_cache()

        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(dataloader)

def evaluate(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module):

        model.eval()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc='Evaluating', leave=False, total=len(dataloader))
        with th.no_grad():
            for i, (src, trg) in enumerate(progress_bar):
                if to_lang == 'en':
                    src, trg = trg, src
                src = src.to(device)
                trg = trg.to(device)
                output = model(src, trg, teacher_forcing_ratio=0)
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            output = model.translate(valid_text)
            print(f'src:\t{" ".join([fr_vocab.get_itos()[idx] for idx in valid_text.squeeze(0).tolist()])}')
            print(f'pred:\t{" ".join([en_vocab.get_itos()[idx] for idx in output])}')
            print(f'trg:\t{valid_label}')

        return epoch_loss / len(dataloader)

def run(model: nn.Module,
        n_epochs: int,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        optimizer: th.optim.Optimizer,
        criterion: nn.Module,
        clip: float):

    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_dataloader, criterion)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            th.save(model.state_dict(), 'model.pt')
        print(f'\nEpoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

        th.cuda.empty_cache()

    return train_losses, valid_losses


history = run(model, 10, train_loader, val_loader, optimizer, criterion, 1)
print(history)