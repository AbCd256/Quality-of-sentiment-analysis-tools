import sys
sys.path.append (".")
import numpy as np
import torch
from char_cnn_utils import load_dataset , create_emb_layer , embedding_data
import torch.nn as nn
import torch.nn.functional as F
import pickle
GPU_AVAILABLE = (torch.cuda.is_available ())
np.random.seed (0)
torch.manual_seed (0)


class char_cnn (nn.Module):
    def __init__(self , char_embedding_dim, WIN_SIZES=[3 , 4 , 5] , NUM_FILTERS=100 , EMBEDDING_DIM=50 , weight_matrix=None ,
                 input_len=None , char_input_len=None, dropout_prob=0.5 , FUNCTION=0 , num_classes=3 , mode="None", char_weight_matrix= None, out_embedding=None, char_win_size=[13,1]):
        super (char_cnn , self).__init__ ()

        print(input_len)
        print(char_input_len)
        self.WIN_SIZES = WIN_SIZES
        self.char_win_size= char_win_size
        self.embedding , _ , _ = create_emb_layer (weight_matrix)
        self.embedding_char , _ , _ = create_emb_layer (char_weight_matrix)
        self.embedding_char.weight.requires_grad= True
        self.embedding.weight.requires_grad = mode == "nonstatic"
        self.input_len = input_len
        self.char_input_len = char_input_len

        if GPU_AVAILABLE:
            self.embedding = self.embedding.cuda ()

        self.conv2d_0= nn.Conv2d(in_channels=char_embedding_dim, out_channels= 5, kernel_size=1,stride=1)
        conv_blocks_chars = []

        #maxpool_kernel_size_char = self.char_input_len - win_size + 1
        self.conv2d = nn.Conv2d (in_channels= 5 , out_channels= out_embedding , kernel_size=char_win_size ,
                                stride=1)






        # create conv blocs
        conv_blocks = []
        for win_size in WIN_SIZES:
            maxpool_kernel_size = self.input_len - win_size + 1
            conv1d = nn.Conv1d (in_channels=(EMBEDDING_DIM+out_embedding) , out_channels=NUM_FILTERS , kernel_size=win_size ,
                                stride=1)
            component = nn.Sequential (
                conv1d ,
                nn.ReLU () ,
                nn.MaxPool1d (kernel_size=maxpool_kernel_size)
            )
            conv_blocks.append(component)

        self.conv_blocks = nn.ModuleList (conv_blocks)
        self.fc = nn.Linear (NUM_FILTERS * len (WIN_SIZES) , num_classes)

    # forward propagation
    def forward(self , x):  # x: (batch, sentence_len)
        x_embedding = self.embedding (x[0])  # embed the input x (batch, max_sentence, len_embedding
        x_embedding_char= self.embedding_char(x[1]) # embed the input x (batch, max_sentence, max_word_men, len_embedding

        #x_embedding_char = x_embedding_char.transpose (2 , 3)  #   convert x to (batch, max_sentence, len_embedding, max_word_len)
        x_embedding_char = x_embedding_char.transpose (1 , 3)


        #for each element in the  batch embeed the character
        embeddings=[]
        #for x_char in x_embedding_char:
        x_char=x_embedding_char
        x_char=self.conv2d_0(x_char)

        x_list_char = self.conv2d(x_char)

        out_char = x_list_char.squeeze ()

        #out_char = torch.cat (out_char , 1)

        #embeddings.append(out_char)
        #the output is the word embedding
        #do the concatination on the batch
        #out_char= torch.cat (out_char , 2)
        #transform to batch size, word embedding, sentence length
        #out_char = out_char.transpose (0 , 2)
        x_embedding = x_embedding.transpose (1 , 2)  #  convert x to (batch, embedding_dim, sentence_len)

        #concatinate the two embeddings
        x= torch.cat((x_embedding, out_char), 1)

        x_list = [conv_block (x) for conv_block in self.conv_blocks]
        out = torch.cat (x_list , 2)

        out = out.view (out.size (0) , -1)
        #featurs
        feature_extracted = out
        out = F.dropout (out , p=0.5 , training=self.training)
        #classification resyls
        return F.softmax (self.fc (out) , dim=1) , feature_extracted


if __name__ == '__main__':


    #    return np.stack (X.values , axis=0) , X_CHAR.values , np.stack (df_dataset["Golden"].values , axis=0)

    X, X_char, Y = load_dataset ("../data/dev/news.csv",
                   "../../../../data/models/GoogleNews-vectors-negative300.bin")
    with open ("../log/weights" , 'rb') as f:
        pretrained_embeddings = pickle.load (f)
    with open ("../log/weights_char" , 'rb') as f:
        pretrained_embeddings_chars = pickle.load (f)


    max_len_sentence= X_char[1].shape[0]
    max_len_char = X_char[1].shape[1]
    cnn= char_cnn (26,weight_matrix= pretrained_embeddings , input_len=max_len_sentence ,char_input_len=max_len_char,char_weight_matrix= pretrained_embeddings_chars, out_embedding=10)
    print ("\n{}\n".format (str (cnn)))
    input=[]
    input.append(torch.from_numpy (np.array(X)).long ())
    input.append (torch.from_numpy(np.array(X_char)).long())
    cnn.forward(input)
