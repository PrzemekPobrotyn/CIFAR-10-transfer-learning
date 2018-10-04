"""
Code below comes from Broad Institute's Keras-Resnet repository:
https://github.com/broadinstitute/keras-resnet/blob/master/keras_resnet/layers/_batch_normalization.py
"""

import keras


class FreezableBatchNormalization(keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """
    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(FreezableBatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        # return super.call, but set training
        return super(FreezableBatchNormalization, self).call(
            training=(not self.freeze), *args, **kwargs)

    def get_config(self):
        config = super(FreezableBatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config
