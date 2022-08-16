import tensorflow_datasets as tfds
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
import torchvision.transforms as T
import tensorflow as tf
from shift.io.dataset.vtab_preprocessors import vtab_required_preprocessors, vtab_to_preprocess, vtab_preprocess_num_classes
from shift.io.dataset.vtab_preprocessors.clevr import _closest_object_preprocess_fn, _count_preprocess_fn
from shift.io.dataset.vtab_preprocessors.kitti import _closest_vehicle_distance_pp
from shift.io.dataset.vtab_preprocessors.dsprites import dsprites_location_fn, dsprites_orientation_fn

preprocessing_map = {
    'kitti': _closest_vehicle_distance_pp,
    'clevr_count': _count_preprocess_fn,
    'clevr_distance': _closest_object_preprocess_fn,
    'dsprites_location': dsprites_location_fn,
    'dsprites_orientation': dsprites_orientation_fn,
}

def get_transform(image_size):
    transforms = T.Compose([
            T.CenterCrop(image_size),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    return transforms

class TensorFlowDatasetGenerator(Dataset):
    """Converts from TFDS to Torch Dataset."""

    def __init__(self, ds, transform=None, ds_name=""):
        """
        Args:
            tfds (tf.Data.Dataset): A tf.Data.Dataset instance, e.g. tfds.load('mnist', split='train')
        """
        self.ds = ds
        self.image = []
        self.label = []
        if ds_name == "":
            (self.image, self.label) = tfds.as_numpy(self.ds)
        
        elif ds_name in ['oxford_iiit_pet', 'dtd', 'diabetic_retinopathy_detection', 'sun397']:
            for batch in self.ds:
                images = batch['image']
                labels = batch['label']
                self.image.append(images.numpy())
                self.label.append(labels.numpy())
        elif ds_name in ['vtab-sna-train']:
            for batch in self.ds:
                images = batch['image']
                images = tf.tile(images, [1, 1, 3])
                labels = batch['label_azimuth']
                self.image.append(images.numpy())
                self.label.append(labels.numpy())
        elif ds_name in ['vtab-snb-train']:
            for batch in self.ds:
                images = batch['image']
                images = tf.tile(images, [1, 1, 3])
                labels = batch['label_elevation']
                self.image.append(images.numpy())
                self.label.append(labels.numpy())
        elif ds_name in list(vtab_to_preprocess.keys()):
            self.ds = ds.map(preprocessing_map[vtab_to_preprocess[ds_name]])
            for batch in self.ds:
                self.image.append(batch[0].numpy())
                self.label.append(batch[1].numpy())
                
        self.transform = transform

    def __len__(self):
        return self.ds.cardinality()

    def __getitem__(self, idx):
        data = np.transpose(self.image[idx], (2, 0, 1))
        label = self.label[idx]
        data = torch.tensor(data)/255
        label = torch.tensor(label).type(torch.LongTensor)
        if self.transform:
            return self.transform(data), label
        else:
            return data, label
        
def construct_dataloader(ds, ds_info, image_size, sample_size=5000, ds_name=""):
    ds = TensorFlowDatasetGenerator(ds, transform=get_transform(image_size),ds_name=ds_name)
    if sample_size > len(ds):
        sample_size = len(ds)
    indices = np.random.choice(list(range(len(ds))), sample_size, replace=False)
    subset = torch.utils.data.Subset(ds,indices)
    # get dataloader
    data_loader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=True, num_workers=0)
    if ds_name in ['smallnorb']:
        return data_loader, ds_info.features["label_azimuth"].num_classes
    if ds_name in list(vtab_to_preprocess.keys()):
        return data_loader, vtab_preprocess_num_classes[vtab_to_preprocess[ds_name]]
    if ds_name in ['vtab-sna-train']:
        return data_loader, ds_info.features["label_azimuth"].num_classes
    if ds_name in ['vtab-snb-train']:
        return data_loader, ds_info.features["label_elevation"].num_classes
    return data_loader, ds_info.features["label"].num_classes