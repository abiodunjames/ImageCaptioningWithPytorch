import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


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
        super().__init__()
        """
       
        There are 3 components in this architecture.
        * Word Embedding layer
        * LSTM layer (Hidden Layer)
        * Linear layer: This maps the hidden layer to the size of output we want (vocab_size)
        
        :param embed_size
        :param hidden_size
        :param vocab_size
        :num_layers
        """
        
        # Embed layer 
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Lstm layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # Hidden layer
        self.fc = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, features, captions):
        # I dropped the last word or else outputs.shape[1] will never be equal to captions.shape[1]
        captions = captions[:, :-1]
        captions = self.embed(captions)
        
        # Combine the feature vectors for both image and captions
        inputs = torch.cat((features.unsqueeze(1), captions), dim = 1)
        
        lstm_outputs, _ = self.lstm(inputs)
        
        outputs = self.fc(lstm_outputs)

        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        """ accepts pre-processed image tensor (inputs) 
        and returns predicted sentence (list of tensor ids of length max_len) """
        
        ids = []
        
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            hiddens = hiddens.squeeze(dim=1)
            outputs = self.fc(hiddens.squeeze(dim=1))
            
            predicted = outputs.max(1)[1]
            ids.append(predicted.item())
                        
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            
        
        return ids
    
    