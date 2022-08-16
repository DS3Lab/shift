import pandas as pd
from loguru import logger

hf_names = pd.read_csv(".cache/hf_name.csv")
ft_records = pd.read_csv(".cache/ftrecord.csv")
logger.info("Length of ft_records: {}".format(len(ft_records)))

# remove HFText models that are not in the hf_names
ft_records = ft_records[ (ft_records["model_identifier"].isin(hf_names["hf_name"])) | (ft_records["framework"] == "HuggingFace") ]
logger.info("Length of ft_records: {}".format(len(ft_records)))

ft_records.to_csv(".cache/ftrecord_2.csv", index=False)