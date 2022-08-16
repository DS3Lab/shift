import pandas as pd
from transformers import AutoModel
from tqdm import tqdm

image_models = pd.read_csv(".cache/shift_models/image_models_with_size.csv")
text_models = pd.read_csv(".cache/shift_models/text_models_with_size.csv")

# find largest
largest_image_model = image_models.sort_values(by="size", ascending=False).head(1)
largest_text_model = text_models.sort_values(by="size", ascending=False).head(1)
print(largest_image_model)
print(largest_text_model)