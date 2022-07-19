import pandas as pd
import requests
df = pd.read_csv('data/huggingface_cv_models.csv')

for idx, row in df.iterrows():
    model_identifier = row['model_identifier']
    image_size = row['image_size']
    batch_size = row['batch_size']
    json_req = {
        "model": {
            "hf_name": model_identifier,
            "required_image_size": {
                "height": image_size,
                "width": image_size
            }
        },
        "info": {
            "batch_size": batch_size,
            "date_added": "2022-07-12",
            "num_params": 100000000,
            "image_size": image_size
        },
        "finetuned": False
    }
    res = requests.post('http://localhost:8001/register_image_model/', json=json_req)