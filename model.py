import torch
import torch.nn as nn
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend 

class Decoder_conv(nn.Module): 
    def __init__(self, nlayers, ninp, nhid, nstride=1, is_attn=False): 

        self.dec = torch.nn.Conv1d(nhid, ninp, 1) 

    def forward(self, input, hidden, context, external_attns=None): 
        pass


class Encoder_conv(nn.Module): 
    def __init__(self, nlayers, ninp, nhid, nstride=1, is_attn=False): 

        super(Encoder_conv, self).__init__() 
 
        self.ninp = ninp
        self.nhid = nhid
        self.nstride = nstride
        self.is_attn = is_attn 

        self.emb = torch.nn.Conv1d(ninp, nhid, 1)

        self.conv = nn.ModuleList() 
        for i in range(nlayers): 
            sub_module = LSTM_conv(i+1, nhid, nstride, is_attn=is_attn) 
            self.conv.append(sub_module) 

    def forward(self, input, hidden, context, external_attns=None): 

        x = self.emb(input).unsqueeze(0) 
        next_context = [] 
        attns = []

        for i, conv in enumerate(self.conv):

            if self.is_attn is True and external_attns is not None: 
                attn = external_attns[i] 
            else:
                attn = None 

            y, c, attn = conv(x, hidden[:i+2,:,:,:], context[:i+1,:,:,:], external_attn=attn) 

            x = torch.cat([x, y.unsqueeze(0)], 0) 

            next_context.append(c) 
            if self.is_attn is True and external_attns is None: 
                attns.append(attn) 

        next_context = torch.stack(next_context, 0)

        if self.is_attn is True and external_attns is None: 
            attns = torch.stack(attns, 0) 
        else:
            attns = None 

        return x[:,:,:,-self.nstride:], next_context[:,:,:,-self.nstride:], attns

class LSTM_conv(nn.Module):
    def __init__(self, nlayers, ndim, nstride=1, is_attn=False):

        super(LSTM_conv, self).__init__()     

        self.nlayers = nlayers
        self.ndim = ndim 
        self.nstride = nstride
        self.is_attn = is_attn 

        if is_attn is True: 
            self.attention = torch.nn.Sequential( 
                torch.nn.Linear(ndim, 1, bias=None),
                torch.nn.Sigmoid()
            )
 
        conv = []
        for i in range(nlayers): 
            conv.append(torch.nn.Conv1d(ndim, ndim*4, nstride)) 

        self.conv = nn.ModuleList(conv) 
        self.recurrent = torch.nn.Linear(ndim, ndim*4) 

    def forward(self, input, hidden, context, external_attn=None): 

        igates = [] 
        for i, conv in enumerate(self.conv): 

            if self.nstride>1:
                padded_input = torch.cat([hidden[i][:,:,1:], input[i]], 2) 
            else: 
                padded_input = input[i]

            igates.append(conv(padded_input)) 
        
        igates = sum(igates) 
        wsz = igates.size(-1)

        hx = hidden[self.nlayers,:,:,self.nstride-1] 
        cx= context[self.nlayers-1,:,:,self.nstride-1] 

        hx_all = [] 
        cx_all = []     
        attn_scores = [] 

        for i in range(wsz): 
            hgate = self.recurrent(hx) 
            state = fusedBackend.LSTMFused.apply
            hx, cx = state(igates[:,:,i], hgate, cx) 

            # attention to hx
            if self.is_attn is True: 
                if external_attn is None: 
                    if i<wsz-1: 
                        attn_score = self.attention(hx) # 10 by 1 
                        hx = hx*attn_score.expand_as(hx) 
                        attn_scores.append(attn_score) 
                else: 
                    # apply external_attn
                    if i>0: 
                        attn = external_attn[:,i-1].unsqueeze(1).expand_as(hx)
                        hx = hx/attn

            hx_all.append(hx) 
            cx_all.append(cx) 
             
        hx_all = torch.stack(hx_all, 2) 
        cx_all = torch.stack(cx_all, 2) 

        if self.is_attn is True and external_attn is None: 
            attn_scores = torch.cat(attn_scores, 1)
        else:
            attn_scores = None 

        return hx_all, cx_all, attn_scores


if __name__ == '__main__':
    
    nbatch = 10
    ninp = 5
    nhid = 4
    wsz = 8 
    nstride = 3
    nlayers = 4

    # make init_hidden 
    prev_hidden = torch.zeros(nlayers+1, nbatch, nhid, nstride).to('cuda') 
    prev_context = torch.zeros(nlayers, nbatch, nhid, nstride).to('cuda') 

    x = torch.randn(nbatch, 1, wsz).to('cuda') 

    model = Encoder_conv(nlayers, 1, nhid, nstride, is_attn=True).to('cuda') 

    h, c, attns = model(x, prev_hidden, prev_context)
    h, c, n_attns = model(x, prev_hidden, prev_context, attns)

    # decoder, teacher forcing + free running ratio control. 
    # long sequence -> decoding. train! 
