from sklearn.utils import shuffle
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from timeit import default_timer
from transformers import AutoModelForImageClassification
from torch.optim.lr_scheduler import MultiStepLR

class HuggingfaceImageTrainer():
    def __init__(self, image_size, device):
        self.image_size = image_size
        self.device = device
    
    def finetune(self, model_name, train_loader, num_classes, lr, epochs, momentum, nesterov):
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        criterion = nn.CrossEntropyLoss()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        sgd_optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov
        )
        scheduler = MultiStepLR(
            sgd_optimizer,
            milestones=[int(epochs/3), int(2*epochs/3)],
            gamma=0.1
        )
        num_training_steps = epochs * len(train_loader)
        progress_bar = tqdm(range(num_training_steps))
        start = default_timer()
        for epoch in range(epochs):
            model.train(True)
            loss_epoch = 0
            batch_count = 0
            progress_bar.set_description(f"Epoch {epoch}")
            for batch in train_loader:
                sgd_optimizer.zero_grad()
                batch[0], batch[1] = batch[0].to(
                    self.device), batch[1].to(self.device)
                outputs = model(batch[0])
                loss = criterion(outputs.logits, batch[1])
                loss.backward()
                loss_epoch += loss.item()
                batch_count += 1
                sgd_optimizer.step()
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=loss.item(), lr=scheduler.get_last_lr())
            average_loss = loss_epoch / batch_count
            self.model = model
        stop = default_timer()
        logger.info(f"Finished training in {stop - start} seconds")

    def test(self, test_ds, batch_size):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,shuffle=True)
        correct = 0
        for batch in test_loader:
            batch[0], batch[1] = batch[0].to(
                self.device), batch[1].to(self.device)
            with torch.no_grad():
                outputs = self.model(batch[0])
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += torch.sum(predictions == batch[1])
        test_accuracy = correct / len(test_ds)
        test_accuracy = test_accuracy.item()
        return test_accuracy