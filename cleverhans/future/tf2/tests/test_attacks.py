import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from cleverhans.devtools.checks import CleverHansTest
from cleverhans.future.tf2.attacks.fast_gradient_method import \
    fast_gradient_method


class TestModel(Model):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = Conv2D(
            3, 1, strides=1, activation="relu", padding="valid")
        self.conv2 = Conv2D(
            6, 1, strides=1, activation="relu", padding="valid")
        self.flatten = Flatten()
        self.dense1 = Dense(5, activation="relu")
        self.dense2 = Dense(2)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


TEST_INPUT_SHAPE = (100, 2, 2, 1)


class TestCommons(CleverHansTest):
    def setUp(self):
        super(TestCommons, self).setUp()
        # fix random seed for consistent results
        tf.random.set_seed(42)

        # setup model and input
        self.model = TestModel()
        self.x = tf.random.uniform(TEST_INPUT_SHAPE)

        # run model once so all variables are created
        self.model(self.x)

        # supported norms
        self.ord_list = [1, 2, np.inf]

    def _model_prediction(self, x):
        return tf.argmax(self.model(x), axis=-1)

    def _attack_success_rate(self, attack_fn, **attack_kwargs):
        x_adv = attack_fn(
            model_fn=self.model, x=self.x, **self.attack_param)

        normal_pred = self._model_prediction(self.x)
        adv_pred = self._model_prediction(x_adv)
        success_rate = tf.reduce_mean(
            tf.cast(tf.equal(normal_pred, adv_pred), tf.float32)
        )

        self.assertLess(success_rate.numpy(), .5)

    def _attack_success_rate_targeted(self, attack_fn, **attack_kwargs):
        # generate random labels to target
        y_target = tf.random.uniform(
            minval=0, maxval=2, shape=self.x.shape[:1], dtype=tf.int64).numpy()

        x_adv = attack_fn(
            model_fn=self.model, x=self.x, y=y_target, targeted=True, **self.attack_param)
        adv_pred = self._model_prediction(x_adv)

        success_rate = tf.reduce_mean(
            tf.cast(tf.equal(y_target, adv_pred), tf.float32)
        )

        self.assertGreater(success_rate.numpy(), .7)

    def _test_for_all_norms(self, test_fn, attack_fn):
        for norm in self.ord_list:
            with self.subTest(norm=norm):
                params = self.attack_param
                params.update({"norm": norm})
                test_fn(
                    attack_fn, **params)


class TestFastGradientMethod(TestCommons):

    def setUp(self):
        super(TestFastGradientMethod, self).setUp()
        self.attack_param = {
            'eps': .5,
            'clip_min': -5,
            'clip_max': 5,
        }

    def test_adv_example_success_rate(self):
        self._test_for_all_norms(
            self._attack_success_rate, fast_gradient_method)

    def test_adv_example_success_rate_targeted_linf(self):
        self._test_for_all_norms(
            self._attack_success_rate_targeted, fast_gradient_method)
