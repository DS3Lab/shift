import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset


class ReaderDataset(Dataset):
    def __init__(self, feature_path, label_path, transform=None, shuffle=True):

        self.feature = torch.load(feature_path)
        self.label = torch.load(label_path)
        if shuffle:
            # shuffle the feature and get the same order of label
            indices = torch.randperm(len(self.feature))
            self.feature = self.feature[indices]
            self.label = self.label[indices]
        self.transform = transform

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.transform is not None:
            return self.transform(self.feature[idx]), self.label[idx]
        return self.feature[idx], self.label[idx]

    @property
    def num_classes(self):
        return self.label.max().item() + 1


class ReaderDatasetGenerator(IterableDataset):
    """Converts from TFDS to Torch Dataset."""

    def __init__(self, feature_path, label_path, transform=None):
        self.feature = torch.load(feature_path)
        self.label = torch.load(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.feature)

    @property
    def num_classes(self):
        return self.label.max().item() + 1

    def __iter__(self):
        for feature, label in zip(self.feature, self.label):
            if self.transform:
                yield self.transform(feature), label
            else:
                yield feature, label
