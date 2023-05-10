import os, wget, subprocess, argparse
import numpy as np
import torch
import torch.optim as optim
import h5py
from dataset import SNLI_data
import train


class Model(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, feature_extractor='attention', use_BN=False, dropout_rate=0.0, POS_embedding=None):
        super(Model, self).__init__()
        # embedding
        self.embedding_dim = embedding_matrix.shape[1]
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.POS_embedding = POS_embedding
        if POS_embedding is not None:
            self.num_POS = POS_embedding.shape[1]
            self.POS_embedding = nn.Embedding.from_pretrained(POS_embedding)    
        else:
            self.num_POS = 0
        
        self.feature_extractor = feature_extractor
        assert feature_extractor in ['attention', 'lstm', 'combine']
        if feature_extractor == 'self_attention':
            # Hopfield pooling
            # self.feature_extractor_module = SelfAttention_Module(hidden_dim, use_BN, dropout_rate)
        else:
            # Sparse Hopfield pooling
            self.feature_extractor_module = LSTM_Module(hidden_dim, dropout_rate=dropout_rate)
        
        self.LSTM = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        
        
        # linear transformation from embedding
        self.input_fc = nn.Linear(self.embedding_dim + self.num_POS, hidden_dim, bias=True)
        # linear transformation to prediction
        self.output_fc = nn.Linear(hidden_dim, 3, bias=True)
        
    def forward(self, a, b, a_POS=None, b_POS=None):
        l_a = a.shape[1]
        l_b = b.shape[1]
        
        a = self.embedding(a) # a: (batch_size x l_a x embedding_dim)
        b = self.embedding(b) # b: (batch_size x l_b x embedding_dim)
        if self.POS_embedding is not None:
            try:
                a_POS = self.POS_embedding(a_POS)
                a = torch.cat((a, a_POS), dim=-1)
                b_POS = self.POS_embedding(b_POS)
                b = torch.cat((b, b_POS), dim=-1)
            except Exception as e:
                print(e)
        
        
        if self.feature_extractor is not 'combine':
            a = self.input_fc(a.view(-1, self.embedding_dim + self.num_POS))
            b = self.input_fc(b.view(-1, self.embedding_dim + self.num_POS))
            a = a.view(-1, l_a, self.hidden_dim) # a: (batch_size x l_a x hidden_dim)
            b = b.view(-1, l_b, self.hidden_dim) # b: (batch_size x l_b x hidden_dim)
        else:
            a, (_, _) = self.LSTM(a)
            b, (_, _) = self.LSTM(b)
            a = a.contiguous().view(-1, l_a, self.hidden_dim)
            b = b.contiguous().view(-1, l_b, self.hidden_dim)
        
        
        h = self.feature_extractor_module(a, b)
        
        y_hat = self.output_fc(h)
        
        return y_hat

def download_data(glove='glove.6B.zip'):
    print("Downloading word embedding...")
    downloaded_glove1 = wget.download("http://nlp.stanford.edu/data/{}".format('glove.6B.zip'))
    downloaded_glove2 = wget.download("http://nlp.stanford.edu/data/{}".format('glove.42B.300.zip'))
    print("Downloading SNLI dataset...")
    downloaded_snli = wget.download("https://nlp.stanford.edu/projects/snli/snli_1.0.zip")
    
    if not os.path.exists("./data"):
        os.mkdir("./data")
    print("Extracting...")
    zip = zipfile.ZipFile(downloaded_glove1)
    zip.extractall(path="./data")
    zip = zipfile.ZipFile(downloaded_glove2)
    zip.extractall(path="./data")
    zip = zipfile.ZipFile(downloaded_snli)
    zip.extractall(path="./data")
    print("done!")


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--download_data', action='store_true', help="download mode")
    parser.add_argument('--data_folder', help="location of folder with the preprocessed data")
    parser.add_argument('--preprocess_data', action='store_true', help="preprocess data")
    parser.add_argument('--embedding', choices=['6B.50d', '6B.100d', '6B.200d', '6B.300d', '42B.300d'], \
                        help="type of word embedding_matrix, one in ['6B.50d', '6B.100d', '6B.200d', '6B.300d', '42B.300d']")
    parser.add_argument('--use_POS', action='store_true', help="use POS tag feature")
    parser.add_argument('--model_type', choices=['attention', 'lstm', 'combine'], default='attention', \
                                        help="type of model to use: ['attention', 'lstm', 'combine']")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden dimension")
    parser.add_argument('--learning_rate', type=float, default=0.0002, help="learning rate")
    parser.add_argument('--dropout_rate', type=float, default=0.2, help="dropout rate")
    parser.add_argument('--max_epochs', type=int, default=100, help="maximum epochs")
    parser.add_argument('--gpu', action='store_true', help="use gpu to train")
    args = parser.parse_args()
    return args
    
    
if __name__=='__main__':
    args = parse_args()
    if args.download_data:
        download_data()
        exit(0)
    
    if args.preprocess_data:
        assert args.embedding is not None
        subprocess.call("python preprocess.py --data_folder=./data/snli_1.0 \
                        --glove=./data/glove.{}.txt --seqlength=100".\
                        format(args.embedding))
        exit(0)
    
    f = h5py.File("./preprocessed/glove.hdf5", 'r')
    wordvec_matrix = torch.from_numpy(np.array(f['word_vecs'], dtype=np.float32))
    
    use_padding = False
    train_data = SNLI_data("./preprocessed/train.hdf5", use_padding=use_padding)
    dev_data = SNLI_data("./preprocessed/dev.hdf5", use_padding=use_padding)
    test_data = SNLI_data("./preprocessed/test.hdf5", use_padding=use_padding)
    
    POS_embedding = None
    if args.use_POS:
        POS_embedding = torch.eye(int(train_data.POS_size.value))
    
    # set the device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Found device: ", device)
        
    model = Model(embedding_matrix=wordvec_matrix, hidden_dim=args.hidden_dim, \
              feature_extractor=args.model_type, dropout_rate=args.dropout_rate, POS_embedding=POS_embedding).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if not os.path.exists("./saved_model"):
        os.mkdir("./saved_model")
    model_path = "./saved_model/model.pt"
    
    train.run(model=model,  \
              optimizer=optimizer, \
              train_data=train_data, \
              dev_data=dev_data, \
              test_data=test_data, \
              max_epochs=args.max_epochs, \
              device=device, \
              model_path=model_path)