import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# method used to import constants
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import constants

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset
    """

    def __init__(self, data_path="/Users/Terru/Desktop/UCLA/ACM AI/Projects/train.csv"):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''

        # Import data
        self.df = pd.read_csv(data_path)

        # Generates embeddings
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

        # TO-DO: possibly stem embeddings!

    # Returns an instance from the dataset
    def __getitem__(self, i):
        '''
        i (int): the desired instance of the dataset
        '''
        # return the ith sample's list of embeddings for each word and label

        text = self.df.iloc[i, 1]

        # basic preprocessing——case, removing punctuation
        text = text.lower().split()
        text = [word.translate(str.maketrans('', '', string.punctuation)) for word in text]

        # lemmatizing
        lemma = WordNetLemmatizer()
        text = [lemma.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]

        # TO-DO: Using a map/dict to remove contractions?

        # Generating embeddings
        embeddings = [self.embedding[i] for i in text[:constants.MAX_SENT_LENGTH] if i in self.embedding]
        if len(embeddings) < constants.MAX_SENT_LENGTH:
            add = [np.zeros(300) for i in range(constants.MAX_SENT_LENGTHlen(embeddings))]
            embeddings.extend(add)

        return embeddings, self.df.iloc[i, 2]

    # Returns the size of the dataset
    def __len__(self):
        return len(self.df)


# data = StartingDataset()
# # print(data.df.head())
# # print(len(data))
# # print(data.embedding["the"])
# print(data[2])
# print(len(data[2][0]))