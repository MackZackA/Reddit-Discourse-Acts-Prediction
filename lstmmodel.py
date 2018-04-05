import nltk
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


torch.manual_seed(1)
# convert it to LSTMText2Word
class LSTMText2Word(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, labelset_size, lstm_layers):
        super(LSTMText2Word, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = labelset_size
        self.lstm_layers = lstm_layers
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.lstm_layers, \
                            bias=False, batch_first=True, dropout=0, bidirectional=False)

        # The linear layer that maps from hidden state space to label space
        self.hidden2label = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sequence):
        '''
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        label_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        label_scores = F.log_softmax(label_space, dim=1)
        return label_scores
        '''
        print("Print sequence and size:")
        print(sequence.size())
        # lstm_out, self.hidden = self.lstm(sequence.view(len(sequence), 1, -1), self.hidden)
        # -1 means inferring from other dimensions
        lstm_out, self.hidden = self.lstm(-1, 1, sequence.view(len(sequence)), None)
        # lstm_out, self.hidden = self.lstm(sequence.view(-1, 1, len(sequence)), None)
        print("\n\n\nPrint after lstm output:")
        print(type(lstm_out))
        print(lstm_out.size())
        label_space = self.hidden2label(lstm_out.view(-1, len(sequence)))
        predicted_label = F.log_softmax(label_space, dim=1)
        print("Print output:")
        print(output)
        print("Print self.hidden:")
        print(self.hidden)
        return output  

def load_json(label_to_index, path):
    '''
    This function processes Felicity's json file.
    For each id, extract a list of tuples, where each tuple consists of (body, majority_type).
    Note that "body" is a list of tokenized, lemmatized, lower-case, no-punctuation words.
    '''
    data_tuple = []
    data = json.load(open(path))
    for id in data:
        if data[id]['body'] != None and data[id]['majority_type'] != None:
            body = data[id]['body']
            # body = text_to_word_list(body)
            majority_type = data[id]['majority_type']
            majority_type = label_to_index[majority_type]
            data_tuple.append((body, majority_type))
    print("Data file is loaded.")
    with open('training_tuples.txt', 'w') as result:
        json.dump(data_tuple, result)
    print("The data tuples are saved as 'training_tuples.txt'.")

def text_to_word_list(text):
    '''
    This function is implemented within load_json().
    It cleans the raw comment from 'body' feature into a list 
    of tokenized, lemmatized, lower-case, no-punctuation words.
    '''
    # word_list = tokenizer.tokenize(text)
    word_list = nltk.word_tokenize(text)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
    return word_list

def load_glove():
    '''
    This function loads the pretrained GloVe vectors trained from CommonCrawl.
    '''
    embeddings = {}
    # p = re.compile('(+|-)*\d+\.\d+')
    with open(os.path.join('/home/zsong/working/data/glove', 'glove.840B.300d.txt')) as glove:
        for line in glove:
            values = line.split()
            # word, vector = values[0], np.asarray(values[1:], dtype='float32')
            word, vector = values[0], values[1:]
            raw_string = ''.join(vector)
            if 'com' in raw_string or '@' in raw_string:
                continue
            if '.' in vector:
                continue
            if 'name@domain.com' in vector:
                continue
            vector = [float(i) for i in vector]
            embeddings[word] = vector
    print("Gloved is loaded.")
    with open('glove.txt', 'w') as result:
        json.dump(embeddings, result)
    print("Glove word vectors are saved as 'glove.txt'.")
         
def prepare_sequence(comment, glove):
    '''
    This function converts a sequence of words into a list of word vectors.
    The output is contained in a PyTorch variable.
    '''
    word_list = text_to_word_list(comment)
    sequence = [get_word_vectors(word, glove) for word in word_list]
    # sequence = np.array(sequence)
    # tensor = torch.from_numpy(sequence)
    tensor = torch.DoubleTensor(sequence)
    return autograd.Variable(tensor)

def get_word_vectors(word, glove):
    '''
    This function retrieves the respective word vector for the word.
    If the word is not in the model, then return a Numpy array of zeros.
    Return: a word vector of 1 * 300.
    '''
    if word in glove:
        # return torch.from_numpy(np.array(glove[word]))
        return glove[word]
    else:
        # return autograd.Variable(torch.from_numpy(np.zeros(300)))
        return [0.0 * 300]

def training(train_data, vectors):
    '''
    This function replicates the training process below.
    There is no return value. The model will be saved with torch.save() and later loaded with torch.load(). 
    '''
    model = LSTMText2Word(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, LAYER_NUM)
    print("Print LSTM architecture:")
    print(model)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    label_scores = 0
    print('Initialize label score', label_scores)
    for epoch in range(100):
        for sequence, label in train_data:
            print("Epoch:", epoch) 
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            print("Starting step 2:")
            # Step 2. Get our inputs ready for the network.
            sequence_input = prepare_sequence(sequence, vectors) 
          
            # Step 3. Run our forward pass.
            print("Starting step 3:")
            print(model(sequence_input))
            print((model(sequence_input)).size())
            predicted_label = model(sequence_input)

            print("Starting step 4:")
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step() 
            loss = loss_function(predicted_label, label) # need to convert predicted label from Variable to a label
            loss.backward()
            optimizer.step()
    print("Training is done.")
    print("The trained perceptron:", optimizer)
    torch.save(model, 'model.pkl')
    print("The model is saved as 'model.pkl'.")

def cross_validation():
    '''
    model = torch.load('model.pkl')
    '''
    pass

data_tuple = []
DATA_PATH = "/home/zsong/working/my_trial_DIR/post_df_parent_text_unified.json" 
vectors = []
label_to_index = {"agreement": 0, "announcement": 1, "answer": 2, "appreciation": 3, "disagreement": 4, "elaboration": 5, "humor": 6, "negativereaction": 7, "question": 8}
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
OUTPUT_DIM = len(label_to_index)
LAYER_NUM = 2

if __name__ == "__main__":
    # load_json(label_to_index, DATA_PATH)
    # load_glove() 
    data_tuple = json.load(open('training_tuples.txt'))
    vectors = json.load(open('glove.txt'))
    training(data_tuple, vectors)
