import os
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from finetune.preprocessing import _TFImageHelper
from loguru import logger
from schemas.mapping.vtab import _VTABMapping
from schemas.models.image_model import TFFullImageModelConfig
from schemas.models.text_model import TFFullTextModelConfig
from schemas.requests.reader import VTABReaderConfig


class VTabFinetuner:
    def __init__(self) -> None:
        self.storage = os.environ["TFHUB_CACHE_DIR"]
        self.vtabmapping = _VTABMapping(include_embed_feature=True, include_label=True)

    def finetune(
        self,
        model,
        finetune_readers: List[VTABReaderConfig],
        hash: str,
        required_image_size,
        num_labels: int = 2,
        learning_rate: float = 0.01,
        epochs: int = 50,
        batch_size: int = 8,
    ):
        is_tf_v1 = False
        if isinstance(model, TFFullTextModelConfig):
            model = hub.load(model.tf_text_model_url)
        elif isinstance(model, TFFullImageModelConfig):
            if "vtab" in model.tf_image_model_url:
                model = model.tf_image_model_url
                is_tf_v1 = True
            else:
                model = hub.load(model.tf_image_model_url)
        else:
            raise ValueError("Unsupported model type {}".format(type(model)))
        self.hash = hash
        vtab_specs = self.vtabmapping.get_specs(finetune_readers[0].vtab_name)
        data_dir = os.environ["TFDS_LOCATION"]
        ds_train, ds_info = tfds.load(
            name=vtab_specs.name,
            split=vtab_specs.splits[finetune_readers[0].split],
            data_dir=data_dir,
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        if required_image_size is not None:
            logger.info("Reshaping images to {}".format(required_image_size))
            ds_train = ds_train.map(
                lambda x, y: (
                    _TFImageHelper.central_crop_with_resize_3_channels(
                        x, (required_image_size.height, required_image_size.width)
                    ),
                    y,
                )
            )
        image_size = required_image_size.height
        num_classes = ds_info.features["label"].num_classes
        ds_train = ds_train.shuffle(10000)
        ds_train = ds_train.batch(batch_size)
        # now construct finetunning model
        if not is_tf_v1:
            m = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)),
                    hub.KerasLayer(model, trainable=True),
                    tf.keras.layers.Dense(
                        num_classes,
                        activation="softmax",
                        kernel_regularizer=tf.keras.regularizers.l2(0.0),
                    ),
                ]
            )
        else:
            raise ValueError("Unsupported TensorFlow v1 model")
        m.build([None, image_size, image_size, 3])
        m.compile(
            optimizer=tf.keras.optimizers.SGD(
                lr=learning_rate, momentum=0.9, nesterov=True
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        m.fit(
            ds_train,
            epochs=epochs,
            verbose=1,
        )
        model_path = os.path.join(self.storage, self.hash)
        logger.info("Saving to ...{}".format(model_path))
        m.save(model_path)
        return model_path
