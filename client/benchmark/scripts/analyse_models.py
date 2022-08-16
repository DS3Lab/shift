import pandas as pd
from transformers import AutoModel
from tqdm import tqdm

image_models = pd.read_csv(".cache/shift_models/image_models.csv")
text_models = pd.read_csv(".cache/shift_models/text_models.csv")

image_models_with_size = []
text_models_with_size = []

for model in tqdm(image_models["model_identifier"]):
    model_instance = AutoModel.from_pretrained(model)
    image_models_with_size.append(
        {
            "model_identifier": model,
            "size": model_instance.num_parameters(),
            "image_size": model_instance.config.image_size,
        }
    )

# for model in tqdm(text_models["model_identifier"]):
#     model_instance = AutoModel.from_pretrained(model)
#     text_models_with_size.append(
#         {
#             "model_identifier": model,
#             "size": model_instance.num_parameters()
#         }
#     )

image_df = pd.DataFrame(image_models_with_size)
text_df = pd.DataFrame(text_models_with_size)
image_df.to_csv(".cache/shift_models/image_models_with_size.csv", index=False)
text_df.to_csv(".cache/shift_models/text_models_with_size.csv", index=False)