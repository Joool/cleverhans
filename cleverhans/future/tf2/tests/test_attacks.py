import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten

from cleverhans.devtools.checks import CleverHansTest
from cleverhans.future.tf2.attacks.fast_gradient_method import \
    fast_gradient_method


class TestModel(Model):
    def __init__(self):
        super(TestModel, self).__init__()
        self.w1 = tf.constant([[1.5, .3], [-2, .3]])
        self.w2 = tf.constant([[-2.4, 1.2], [.5, -2.3]])

    def call(self, x):
        x = tf.linalg.matmul(x, self.w1)
        x = tf.math.sigmoid(x)
        x = tf.linalg.matmul(x, self.w2)
        return x


TEST_INPUT_SHAPE = (100, 2)


class TestCommons(CleverHansTest):
    def setUp(self):
        super(TestCommons, self).setUp()
        # since most attacks are wrapped in tf.function sharing models across tests is
        # diffcult, thus, we recreate the model in each test
        self.model = TestModel()
        self.x = tf.random.uniform(TEST_INPUT_SHAPE)
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
            minval=0, maxval=2, shape=self.x.shape[:1], dtype=tf.int64)

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
