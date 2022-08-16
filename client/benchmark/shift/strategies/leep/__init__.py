import torch
from shift.strategies._base import BaseSearchStrategy
from shift.io.dataset.data_loader import TensorFlowDatasetGenerator, construct_dataloader
from shift.tasks.mapping import ds_to_vtab_mapping, shift_to_aftp_map
from transformers import AutoModel, AutoModelForImageClassification
from loguru import logger
import pandas as pd
from tqdm import tqdm
import requests
import tensorflow_datasets as tfds
import tensorflow as tf
import traceback
import sys
import os

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    details = tf.config.experimental.get_device_details(gpu_devices[0])
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

def notice(msg):
    print(msg)

def leep(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    number_of_target_labels: int,
    device: torch.device,
) -> float:
    """
    data_loader should return pairs of (images, labels), where labels are classes of
    the images, represented as zero-indexed integer
    :param model: Pytorch multi-class model
    :param data_loader: DataLoader for the downstream dataset
    :param number_of_target_labels: The number of the downstream dataset classes
    :param device: Device to run on
    :returns: LEEP score
    :rtype: float
    """

    model.to(device).eval()

    with torch.no_grad():

        # Actual dataset length can be smaller if it's not divisable by batch_size - this is used for tensors pre-allocation
        predicted_dataset_length = len(data_loader) * data_loader.batch_size

        # Get number of upstream dataset classes
        original_output = model(next(iter(data_loader))[
                                      0].to(device=device))
        original_output = original_output['logits']
        original_output_shape = original_output.shape[1]

        # Allocate empty arrays ahead of time

        # Omega from Eq(1) and Eq(2)
        categorical_probability = torch.zeros(
            (predicted_dataset_length, original_output_shape), dtype=torch.float32, device=device)

        all_labels = torch.zeros(
            predicted_dataset_length, dtype=torch.int64, device=device)

        # Joint porbability from Eq (1)
        p_target_label_and_source_distribution = torch.zeros(
            number_of_target_labels, original_output_shape, device=device)

        soft_max = torch.nn.LogSoftmax()

        # This calculates actual dataset length
        actual_dataset_length = 0

        for i, (images, labels) in enumerate(data_loader):
            current_batch_length = labels.shape[0]
            actual_dataset_length += current_batch_length

            images = images.to(device)
            labels = labels.to(device)
            result = model(images)
            result = result['logits']
            # Change to probability
            result = torch.exp(soft_max(result))
            categorical_probability[i*data_loader.batch_size:i *data_loader.batch_size + current_batch_length] = result
            all_labels[i*data_loader.batch_size:i *
                       data_loader.batch_size + current_batch_length] = labels
            p_target_label_and_source_distribution[labels] += result.squeeze()
        # Shrink tensors to actually fit to the actual dataset length
        categorical_probability = torch.narrow(
            categorical_probability, dim=0, start=0, length=actual_dataset_length)
        all_labels = torch.narrow(
            all_labels, dim=0, start=0, length=actual_dataset_length)

        p_target_label_and_source_distribution /= actual_dataset_length
        p_marginal_z_distribution = torch.sum(
            p_target_label_and_source_distribution, axis=0)
        p_empirical_conditional_distribution = torch.div(
            p_target_label_and_source_distribution, p_marginal_z_distribution)

        total_sum = torch.sum(torch.log(torch.sum(
            (p_empirical_conditional_distribution[all_labels] * categorical_probability), axis=1)))
        return (total_sum / actual_dataset_length).item()

def calc_or_fetch_leep_score(model, ds, ds_info, df, ds_name):
    if df is None:
        model_instance = AutoModelForImageClassification.from_pretrained(model)
        image_size = model_instance.config.image_size
        data_loader, num_classes = construct_dataloader(ds, ds_info, image_size, ds_name=ds_name)
        leep_score = leep(model_instance, data_loader, num_classes, torch.device("cuda"))
        return leep_score
    known_models = set(df['model'].values.tolist())
    if model not in known_models:
        model_instance = AutoModelForImageClassification.from_pretrained(model)
        image_size = model_instance.config.image_size
        data_loader, num_classes = construct_dataloader(ds, ds_info, image_size, ds_name=ds_name)
        leep_score = leep(model_instance, data_loader, num_classes, torch.device("cuda"))
        return leep_score
    else:
        logger.info(f"Model {model} already in database")
        
        subset = df[df['model'] == model]
        return subset['score'].values[0]

class LeepSearchStrategy(BaseSearchStrategy):
    """
    LEEP Score as a way of ordering models
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.name=f"leep"
        self.leep_config = config

    def cache_search(self):
        task = self.target['name']
        ds_name = task
        if task.startswith("vtab"):
            ds_name = shift_to_aftp_map[task]['name']
            
        ds, ds_info = tfds.load(ds_name, split='train', with_info=True, as_supervised=False)
        # ds = ds.shuffle(10000)
        ds = ds.take(10000)
        model = self.candidate_models
        scores = []
        for model in tqdm(self.candidate_models):
            try:
                df = None
                if os.path.exists(f".cache/leep/{task}.csv"):
                    df = pd.read_csv(f".cache/leep/{task}.csv")
                leep_score = calc_or_fetch_leep_score(model, ds, ds_info,df, ds_name=task)
                logger.info(f"{model} has LEEP score {leep_score}")
                scores.append({
                    'model': model,
                    'score': leep_score,
                    'type': 'leep',
                    'task': task
                })
                notice(f"[{task}] {model} has a leep score: {leep_score}")
                scores = pd.DataFrame(scores)
                scores.to_csv(f".cache/leep/{task}.csv", index=False)
                scores = scores.to_dict('records')
            except Exception as e:
                traceback.print_exception(*sys.exc_info())
                logger.error(f"{model} failed with {str(e)}")
                notice("failed")

    def search(self):
        
        self.cache_search()
        task = self.config['target']['name']
        df = pd.read_csv(f".cache/leep/{task}.csv")
        # find the largest leep score and its model_identifier
        sorted_df = df.sort_values(by='score', ascending=False)
        interested_df = sorted_df[:1]
        models = interested_df['model'].values.tolist()
        return models, 5400