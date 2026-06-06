import tf_keras as keras
import numpy as np
from PIL import Image
import tensorflow as tf
from tf_keras import layers
import yaml


class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


class OCRRecognizer:
    CHARACTERS = sorted([
        'E', 'C', 'N', 'p', 'J', 'Y', 'M', 'e', 'g', 'b', 'V', 'G', 'P',
        't', 'U', '3', 'n', 'L', '0', 'B', 'A', '8', 'F', 'O', '2', 'a',
        '/', '1', 'c', 'o', 'y', 'v', 'r', '6', 'R', 'D', '5', 'T', '9',
        'S', 'l', 'u', '7', '4',
    ])

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["model"]["ocr"]

        self.img_width = cfg["img_width"]
        self.img_height = cfg["img_height"]
        self.max_length = cfg["max_length"]

        self.char_to_num = layers.StringLookup(
            vocabulary=self.CHARACTERS, mask_token=None
        )
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

        with keras.utils.custom_object_scope({"CTCLayer": CTCLayer}):
            full_model = keras.models.load_model(cfg["weights"])

        self.model = keras.Model(
            inputs=full_model.get_layer("image").input,
            outputs=full_model.get_layer("dense2").output,
        )

    def _decode_prediction(self, pred_label):
        input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]
        decode = keras.backend.ctc_decode(
            pred_label, input_length=input_len, greedy=True
        )[0][0][:, :self.max_length]

        chars = self.num_to_char(decode)
        texts = [
            tf.strings.reduce_join(inputs=char).numpy().decode("UTF-8")
            for char in chars
        ]
        return [text.replace("[UNK]", " ").strip() for text in texts]

    def recognize(self, image):
        """Read date text from a cropped region.

        Args:
            image: a PIL.Image, or a path to an image file.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image_arr = np.array(image)
        rgb = tf.image.convert_image_dtype(image_arr, tf.float32)[..., :3]
        gray = tf.image.rgb_to_grayscale(rgb)
        gray = tf.transpose(gray, perm=[1, 0, 2])
        resized = tf.image.resize(gray, (self.img_width, self.img_height))

        pred = self.model.predict(tf.expand_dims(resized, axis=0))
        label = self._decode_prediction(pred)[0]
        return str(label).upper()
