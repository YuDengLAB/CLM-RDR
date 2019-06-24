import numpy as np
import tensorflow as tf
from math import sqrt
import time
import datetime
import sys
import os
sys.path.append("./conf")
import config

class Data(object):
    
    def __init__(self,
                 data_source,
                 alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
	             l0 = 150,
	             batch_size = 128,
                 no_of_classes=5):

        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        self.no_of_classes = no_of_classes
        for i, c in enumerate(self.alphabet):
            self.dict[c] = i + 1

        
        self.length = l0
        self.batch_size = batch_size
        self.data_source = data_source

    def loadData(self):
        data = []
        row0 = []
        row1 = []
        row2 = ''
        with open(self.data_source, 'r') as rdr:
            # rdr = csv.reader(f, delimiter=',')
            i = 0
            for row in rdr:
                row1 = row.strip("\n").split(",")
                if len(row1) == 1:
                    row1.append('0')
                row0 = list(row1[0])
                row2 = ' '.join(row0)
                # print(row2)
                data.append((int(row1[1]), row2))
        self.data = np.array(data)
        self.shuffled_data = self.data
        # print(data)

    def load_Test_Data(self):
        data = []
        row0 = []
        row1 = []
        row2 = ''
        with open(self.data_source, 'r') as rdr:
            # rdr = csv.reader(f, delimiter=',')
            i = 0
            for row in rdr:
                row1 = row.strip("\n").split(",")
                if len(row1) == 1:
                    row1.append('0')
                row0 = list(row1[0])
                row2 = ' '.join(row0)
                # print(row2)
                data.append(row2)
        self.data = np.array(data)


    def shuffleData(self):
        data_size = len(self.data)
        
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.shuffled_data = self.data[shuffle_indices]         

    # def getBatch(self, batch_num=0):
    #     data_size = len(self.data)
    #     start_index = batch_num * self.batch_size
    #     end_index = min((batch_num + 1) * self.batch_size, data_size)
    #     return self.shuffled_data[start_index:end_index]

    def getBatchToIndices(self, batch_num=0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = data_size if self.batch_size == 0 else min((batch_num + 1) * self.batch_size, data_size)
        batch_texts = self.shuffled_data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])

        return np.asarray(batch_indices, dtype='int64'), classes

    def strToIndexs(self, s):
        s = s.lower()
        m = len(s)
        n = min(m, self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, n):
            c = s[i]
            if c in self.dict:
                str2idx[i] = self.dict[c]
        return str2idx

    def getLength(self):
        return len(self.data)

    def dev_get(self):
        dev_data = self.shuffled_data
        classes = []
        dev_ = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        for c, s in dev_data:
            dev_.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])
        return np.asarray(dev_, dtype='int64'), classes

    def test_get(self):
        test_data = self.data
        test_ = []
        for s in test_data:
            test_.append(self.strToIndexs(s))
        return np.asarray(test_, dtype='int64')
# if __name__ == '__main__':
#     data = Data("./data/train.csv")
##    E = np.eye(4)
##    img = np.zeros((4, 15))
##    idxs = data.strToIndexs('aghgbccdahbaml')
##    print idxs
    
    # data.loadData()



