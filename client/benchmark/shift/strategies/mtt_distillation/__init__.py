
import pandas as pd

if __name__=="__main__":
    from shift.modules.ftapp.trainer.torch_trainer import HuggingfaceImageTrainer
    from shift.modules.ftapp.dataloader.pt_loader import ReaderDatasetGenerator, ReaderDataset
    from shift.modules.ftapp.dataloader.preprocessing.ImageResize import get_image_resize_transformers
    from shift.modules.ftapp.helper.search import Searcher
    from tqdm import tqdm
    all_models = pd.read_csv(".cache/shift_models/image_models_with_size.csv")
    models = []
    batch_size = 8
    all_results = []
    for idx,row in tqdm(all_models.iterrows()):
        model_name = row['model_identifier']
        image_size = row['image_size']
        ds = ReaderDataset(
            feature_path='.cache/distillation/cifar100/images_best.pt',
            label_path='.cache/distillation/cifar100/labels_best.pt',
            transform=get_image_resize_transformers(image_size),
            shuffle=True,
        )
        searcher = Searcher(ds, test_ratio=0.8)
        models = [{
            'hf_name': model_name,
            'required_image_size': {
                'height': image_size,
                'width': image_size,
            }
        }]
        results = searcher.rank(models)
        all_results.extend(results)
        df = pd.DataFrame(all_results)
        df.to_csv(".cache/distillation/image_models_with_size_results.csv", index=False)