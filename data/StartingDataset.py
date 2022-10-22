import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class StartingDataset(torch.utils.data.Dataset):
    """
    Bag of Words Dataset
    """

    # TODO: dataset constructor.
    def __init__(self, data_path="/Users/Terru/Desktop/UCLA/ACM AI/Projects/train.csv"):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''

        # Preprocess the data. These are just library function calls so it's here for you
        self.df = pd.read_csv(data_path)

        # Embeddings
        self.words = []
        self.word2idx = {}
        self.embedding = {}
        self.idx = 0

        with open("/Users/Terru/Desktop/UCLA/ACM AI/Projects/glove.6B/glove.6B.300d.txt") as f:
            for l in f:
                line = l.split()
                word = line[0]
                vector = np.array([float(number) for number in line[1:]])
                self.embedding[word] = vector
                self.words.append(word)

    # TODO: return an instance from the dataset
    def __getitem__(self, i):
        '''
        i (int): the desired instance of the dataset
        '''
        # return the ith sample's list of word counts and label
        return self.sequences[i, :].toarray(), self.labels[i]

    # TODO: return the size of the dataset
    def __len__(self):
        return self.sequences.shape[0]


data = StartingDataset()
print(data.df.head())
print(len(data.df))
print(data.embedding["the"])