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
                nn.BatchNorm2d(self.n_conv_filters * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (self.n_conv_filters*4) x 8 x 8
                nn.Conv2d(self.n_conv_filters * 4, self.n_conv_filters * 8, 4, 2, 1, bias=False),
            )
            self.image_fc = nn.Linear(self.n_conv_filters * 8 * 4 * 4, self.n_bottleneck)
        else:
            self.image_fc = nn.Linear(self.n_pretrain_image, self.n_bottleneck)

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
            self.text_fc = nn.Linear(self.n_pretrain_text, self.n_bottleneck)

        self.joint_fc = nn.Sequential(
            nn.Linear(self.n_bottleneck * 2, self.n_bottleneck),
            nn.LeakyReLU(),
            nn.Linear(self.n_bottleneck, self.n_bottleneck // 2),
            nn.LeakyReLU(),
            nn.Linear(self.n_bottleneck // 2, 1),
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
        
        if self.train_text_from_scratch:
            batch_size = text_seq.size(0)
            
            if batch_size > 1:
                sorted_len, sorted_idx = torch.sort(text_len, descending=True)
            
            text_seq = text_seq[sorted_idx]
            text_emb = self.text_embed(text_seq)
            text_packed = rnn_utils.pack_padded_sequence(
                text_emb,
                sorted_len.data.tolist() if batch_size > 1 else text_len.data.tolist(), 
                batch_first=True,
            )
            _, text_hid = self.text_gru(text_packed)
            text_hid = text_hid.view(batch_size, -1)

            if batch_size > 1:
                _, reversed_idx = torch.sort(sorted_idx)
            
            text_hid = text_hid[reversed_idx]
            text_hid = self.text_fc(text_hid)
        else:
            assert text_emb is not None
            text_hid = self.text_fc(text_emb)

        concat = torch.cat((image_hid, text_hid), 1)
        return self.joint_fc(concat)
