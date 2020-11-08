import torch
import torch.nn as nn
import torch.nn.functional as F

dim_enc = 512
dim_freq = 80
dim_f0 = 257
num_grp = 32
dim_dec = 512

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
    
    


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        #self.dropout = nn.Dropout(0.0)
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_freq+dim_emb if i==0 else dim_enc,
                         dim_enc,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(num_grp, dim_enc))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(dim_enc, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
                
        for conv in self.convolutions:
            #x = self.dropout(F.relu(conv(x)))
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        #self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes
      
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm = nn.LSTM(dim_neck*2+dim_emb+dim_f0, dim_dec, 3, batch_first=True)
        
        self.linear_projection = LinearNorm(dim_dec, dim_freq)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        
        outputs, _ = self.lstm(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        #self.dropout = nn.Dropout(0.0)
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(dim_freq, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.GroupNorm(num_grp, 512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.GroupNorm(num_grp, 512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, dim_freq,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.GroupNorm(5, dim_freq))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            #x = self.dropout(torch.tanh(self.convolutions[i](x)))
            x = torch.tanh(self.convolutions[i](x))

        #x = self.dropout(self.convolutions[-1](x))
        x = self.convolutions[-1](x)

        return x    
    

    
    
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()
        self.freq = freq


    def forward(self, x, c_org, f0_org=None, c_trg=None, f0_trg=None, enc_on=False):
        
        x = x.transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
                
        codes = self.encoder(x)
        if enc_on:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,self.freq,-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, 
                                     c_trg.unsqueeze(1).expand(-1,x.size(-1),-1),
                                     f0_trg), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)
    
    
    
    
    
