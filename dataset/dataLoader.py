from torch.utils.data import DataLoader, Dataset
from dataset.utils import word_split
import scipy.io as io
from sklearn.model_selection import train_test_split


class Data(Dataset):
    def __init__(self, train=True):
        d = io.loadmat('dataset/data/w2v.mat')
        # x, y = d['x'], d['y']
        self.n_classes = len(set(d['y'].squeeze()))
        X_train, X_test, y_train, y_test, length_train, length_test = train_test_split(d['x'], d['y'].squeeze()-1,
                                                                                       d['length'].squeeze(),
                                                                                       test_size=0.3)
        if train:
            self.x = X_train
            self.y = y_train
            self.length = length_train
        else:
            self.x = X_test
            self.y = y_test
            self.length = length_test

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.length[item]

    def __len__(self):
        return len(self.x)


class DL:
    def __init__(self):
        train_data = Data()
        test_data = Data(False)
        self.n_classes = train_data.n_classes
        self.traindl = DataLoader(train_data, shuffle=True, batch_size=200)
        self.testdl = DataLoader(test_data, shuffle=True, batch_size=200)
