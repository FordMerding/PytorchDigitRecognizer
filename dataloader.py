import numpy as np
from torch.utils.data import Dataset

class MNISTData(Dataset):
    def __init__(self, path, train = True):
        self.path = path
        self.images = []
        self.labels = []
        if(train):
            self.images = self.load_data(path + 'train-images-ubyte', 60000)
            self.labels = self.load_labels(path + 'train-labels-ubyte', 60000)
        else:
            self.images = self.load_data(path + 'test-images-ubyte', 10000)
            self.labels = self.load_labels(path + 'test-labels-ubyte', 10000)
    def load_data(self, file, num_images):
        with open(file, 'rb') as f:
            f.read(16)
            buf = f.read(28 * 28 * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(num_images, 28, 28)
        return data
    def load_labels(self, file, num_labels):
        with open(file, 'rb') as f:
            f.read(8)
            buf = f.read(num_labels)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx]) 