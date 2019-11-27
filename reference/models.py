import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.utils.rnn as rnn_utils


class Supervised(nn.Module):

    def __init__(
            self,
            # ---
            train_image_from_scratch = True,
            train_text_from_scratch = True,
            # ---
            n_pretrain_image = 256,
            n_pretrain_text = 256,
            # ---
            n_bottleneck = 64,
            n_image_channels = 3,
            n_conv_filters = 64,
            vocab_size = None,
            n_embedding = 32,
            n_gru_hidden = 64,
            gru_bidirectional = False,
            n_gru_layers = 1,
        ):
        
        super(Supervised, self).__init__()

        self.train_image_from_scratch = train_image_from_scratch
        self.train_text_from_scratch = train_text_from_scratch
        self.n_pretrain_image = n_pretrain_image
        self.n_pretrain_text = n_pretrain_text
        self.n_bottleneck = n_bottleneck
        self.n_image_channels = n_image_channels
        self.n_conv_filters = n_conv_filters
        self.vocab_size = vocab_size
        self.n_embedding = n_embedding
        self.n_gru_hidden = n_gru_hidden
        self.gru_bidirectional = gru_bidirectional
        self.n_gru_layers = n_gru_layers

        if self.train_image_from_scratch:
            self.image_conv = nn.Sequential(
                # input is (self.n_image_channels) x 64 x 64
                nn.Conv2d(self.n_image_channels, self.n_conv_filters, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (self.n_conv_filters) x 32 x 32
                nn.Conv2d(self.n_conv_filters, self.n_conv_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.n_conv_filters * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (self.n_conv_filters*2) x 16 x 16
                nn.Conv2d(self.n_conv_filters * 2, self.n_conv_filters * 4, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(self.n_conv_filters * 4),
                # nn.LeakyReLU(0.2, inplace=True),
                # state size. (self.n_conv_filters*4) x 8 x 8
                # nn.Conv2d(self.n_conv_filters * 4, self.n_conv_filters * 8, 4, 2, 1, bias=False),
            )
            self.image_fc = nn.Linear(self.n_conv_filters * 4 * 4 * 4, self.n_bottleneck)
        else:
            self.image_fc = nn.Sequential(
                nn.Linear(self.n_pretrain_image, self.n_bottleneck),
                nn.LeakyReLU(),
                nn.Linear(self.n_bottleneck, self.n_bottleneck),
            )

        self.image_to_gru = nn.Sequential(
            nn.Linear(self.n_bottleneck, self.n_gru_hidden),
            nn.ReLU(),
            nn.Linear(self.n_gru_hidden, self.n_gru_hidden),
        )

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        if self.train_text_from_scratch:
            self.text_embed = nn.Embedding(self.vocab_size, self.n_embedding)
            self.text_gru = nn.GRU(
                self.n_embedding, 
                self.n_gru_hidden, 
                batch_first = True, 
                bidirectional = self.gru_bidirectional,
                num_layers = self.n_gru_layers
            )
            n_gru_effect_hidden = self.n_gru_hidden * self.n_gru_layers
            if self.gru_bidirectional:
                n_gru_effect_hidden *= 2
            self.text_fc = nn.Linear(n_gru_effect_hidden, self.n_bottleneck)
        else:
            self.text_fc = nn.Sequential(
                nn.Linear(self.n_pertrain_text, self.n_bottleneck),
                nn.LeakyReLU(),
                nn.Linear(self.n_bottleneck, self.n_bottleneck),
            )

        # https://arxiv.org/pdf/1905.02925.pdf
        self.joint_fc = nn.Sequential(
            nn.Linear(self.n_bottleneck * 2, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(
            self, 
            image, 
            text_seq, 
            text_len, 
            image_emb = None,
            text_emb = None,
        ):

        if self.train_image_from_scratch:
            batch_size = image.size(0)
            image_hid = self.image_conv(image)
            image_hid = image_hid.view(batch_size, -1)
            image_hid = self.image_fc(image_hid)
        else:
            assert image_emb is not None
            image_hid = self.image_fc(image_emb)

        text_gru_h0 = self.image_to_gru(image_hid)
        text_gru_h0 = self.dropout1(text_gru_h0)
        
        if self.train_text_from_scratch:
            batch_size = text_seq.size(0)
            sorted_len, sorted_idx = torch.sort(text_len, descending=True)
            
            text_seq = text_seq[sorted_idx]
            text_emb = self.text_embed(text_seq)
            text_packed = rnn_utils.pack_padded_sequence(
                text_emb,
                sorted_len.data.tolist(),
                batch_first=True,
            )
            _, text_hid = self.text_gru(text_packed, text_gru_h0.unsqueeze(0))
            text_hid = text_hid.view(batch_size, -1)

            _, reversed_idx = torch.sort(sorted_idx)
            
            text_hid = text_hid[reversed_idx]
            text_hid = self.text_fc(text_hid)
        else:
            assert text_emb is not None
            text_hid = self.text_fc(text_emb)

        concat = torch.cat((image_hid, text_hid), 1)
        concat = self.dropout2(concat)
        return self.joint_fc(concat)


class TextImageCompatibility(nn.Module):
    
    def __init__(
            self, 
            vocab_size, 
            img_size = 32, 
            channels = 3, 
            embedding_dim = 64, 
            hidden_dim = 256,
            n_filters = 64,
        ):
        
        super(TextImageCompatibility, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

        self.hidden_dim = 256

        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.txt_lin = nn.Linear(self.hidden_dim, self.hidden_dim // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(channels, n_filters, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters * 2, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters * 2, n_filters * 4, 2, 2, padding=0),
        )

        cout = gen_32_conv_output_dim(img_size)
        self.fc = nn.Linear(n_filters * 4 * cout**2, hidden_dim // 2)
        self.cout = cout
        self.n_filters = n_filters
        self.sequential = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 3), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 3, self.hidden_dim // 9),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 9, self.hidden_dim // 27),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 27, 1),
        )

    def forward(self, img, seq, length):
        assert img.size(0) == seq.size(0)
        batch_size = img.size(0)

        # CNN portion for image
        out = self.conv(img)
        out = out.view(batch_size, self.n_filters * 4 * self.cout**2)
        img_hidden = self.fc(out)

        # RNN portion for text
        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # embed sequences
        embed_seq = self.embedding(seq)

        # pack padded sequences
        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist(), batch_first=True)

        # forward RNN
        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]
        
        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]
        txt_hidden = self.txt_lin(hidden)

        # concat then forward
        concat = torch.cat((txt_hidden, img_hidden), 1)

        return self.sequential(concat)


def gen_32_conv_output_dim(s):
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    return s


def get_conv_output_dim(I, K, P, S):
    # I = input height/length
    # K = filter size
    # P = padding
    # S = stride
    # O = output height/length
    O = (I - K + 2*P)/float(S) + 1
    return int(O)

