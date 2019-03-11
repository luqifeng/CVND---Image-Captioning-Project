import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        captions = self.embeddings(captions)
        embed = torch.cat((features.unsqueeze(1),captions),1)
        r_out = self.lstm(embed)
        output = self.linear(r_out[0])[:, :-1, :]
        return output

    def sample(self, inputs, states=None, max_len=20):
        #" accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #pass
        output = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            mid = self.linear(hiddens.squeeze(1))
            predicted = mid.max(1)[1]
            output.append(predicted.tolist()[0])

            inputs = self.embeddings(predicted)
            inputs = inputs.unsqueeze(1) 
        #print(output)
        #output = torch.cat(output, 1)
        return output