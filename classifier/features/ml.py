import skops
import tensorflow as tf

from features.core import BaseFeature

FML01_MODEL_FILEPATH = 'classifier/molels/text_mlp_model'
FML02_MODEL_FILEPATH = 'classifier/molels/impersonation_md_2204'


class FML01(BaseFeature):
    DESCRIPTION = "Email text was classified as spam by ML01 model"
    BLOCKED_BY = []

    def init(self):
        (
            self.vectorizer,
            self.model
        ) = tf.keras.models.load_model(FML01_MODEL_FILEPATH)


    def check(self):
        vector = self.vectorizer[0].transform(self.envelope.text)
        vector = self.vectorizer[1].transform(vector).toarray()
        target = self.model.predict(vector)

        return target


class FML02(BaseFeature):
    DESCRIPTION = "Email was classified as impersonation by ML02 model"
    BLOCKED_BY = []

    def init(self):
        with open(FML02_MODEL_FILEPATH, 'rb') as f:
            self.model = skops.io.loads(f.read())

    def check(self):
        target = self.model.predict(self.envelope.features["header"])

        return target
