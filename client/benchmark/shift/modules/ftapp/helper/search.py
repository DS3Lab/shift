from shift.modules.ftapp.dataloader.pt_loader import ReaderDataset
from shift.modules.ftapp.trainer.torch_trainer import HuggingfaceImageTrainer
from shift.modules.ftapp.dataloader.preprocessing.ImageResize import get_image_resize_transformers
from torch.utils.data import Subset
import torch
class Searcher():
    def __init__(self, dataset: ReaderDataset, test_ratio=0.8) -> None:
        self.ds = dataset
        self.train_ds = Subset(self.ds, range(0, int(len(self.ds) * test_ratio)))
        self.test_ds = Subset(self.ds, range(int(len(self.ds) * test_ratio), len(self.ds)))

    def rank(self, model_specs):
        results = []
        for spec in model_specs:
            try:
                batch_size = 8
                model_name = spec['hf_name']
                required_size = spec['required_image_size']['height']
                trainer = HuggingfaceImageTrainer(image_size=required_size, device='cuda')
                train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(self.test_ds, batch_size=batch_size, shuffle=True)
                trainer.finetune(
                    model_name=model_name,
                    train_loader = train_loader,
                    num_classes=self.ds.num_classes,
                    lr=0.01,
                    epochs=5,
                    momentum=0.9,
                    nesterov=True,
                )
                test_accuracy = trainer.test(self.test_ds, batch_size)
                results.append({
                    'name': model_name,
                    'score': test_accuracy,
                })
            except Exception as e:
                print(e)
                continue
        return results