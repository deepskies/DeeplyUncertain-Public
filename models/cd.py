import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import InputSpec, Dense, Wrapper, Input, concatenate
from tensorflow.keras.models import Model
import numpy as np


class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=0, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()

        # initialise p
        self.p_logit = self.add_weight(name='p_logit',
                                       shape=(1,),
                                       initializer=tf.random_uniform_initializer(self.init_min, self.init_max),
                                       dtype=tf.dtypes.float32,
                                       trainable=True)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x, p):
        """
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        """
        eps = 1e-07
        temp = 0.1

        unif_noise = tf.random.uniform(shape=tf.shape(x))
        drop_prob = (
            tf.math.log(p + eps)
            - tf.math.log(1. - p + eps)
            + tf.math.log(unif_noise + eps)
            - tf.math.log(1. - unif_noise + eps)
        )
        drop_prob = tf.math.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        p = tf.math.sigmoid(self.p_logit)

        # initialise regulariser / prior KL term
        input_dim = inputs.shape[-1]  # last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1. - p)
        dropout_regularizer = p * tf.math.log(p) + (1. - p) * tf.math.log(1. - p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs, p)), regularizer
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs, p)), regularizer

            return tf.keras.backend.in_train_phase(relaxed_dropped_inputs,
                                                   self.layer.call(inputs),
                                                   training=training), regularizer


def mse_loss(true, pred):
    n_outputs = pred.shape[1] // 2
    mean = pred[:, :n_outputs]
    return tf.reduce_mean((true - mean) ** 2, -1)


def heteroscedastic_loss(true, pred):
    n_outputs = pred.shape[1] // 2
    mean = pred[:, :n_outputs]
    log_var = pred[:, n_outputs:]
    precision = tf.math.exp(-log_var)
    return tf.reduce_sum(precision * (true - mean) ** 2. + log_var, -1)


def make_model(n_features, n_outputs, n_nodes=100, dropout_reg=1e-5, wd=0):
    losses = []
    inp = Input(shape=(n_features,))
    x = inp
    x, loss = ConcreteDropout(Dense(n_nodes, activation='relu'),
                              weight_regularizer=wd, dropout_regularizer=dropout_reg)(x)
    losses.append(loss)
    x, loss = ConcreteDropout(Dense(n_nodes, activation='relu'),
                              weight_regularizer=wd, dropout_regularizer=dropout_reg)(x)
    losses.append(loss)
    x, loss = ConcreteDropout(Dense(n_nodes, activation='relu'),
                              weight_regularizer=wd, dropout_regularizer=dropout_reg)(x)
    losses.append(loss)
    mean, loss = ConcreteDropout(Dense(n_outputs), weight_regularizer=wd, dropout_regularizer=dropout_reg)(x)
    losses.append(loss)
    log_var, loss = ConcreteDropout(Dense(n_outputs), weight_regularizer=wd, dropout_regularizer=dropout_reg)(x)
    losses.append(loss)
    out = concatenate([mean, log_var])
    model = Model(inp, out)
    for loss in losses:
        model.add_loss(loss)

    model.compile(optimizer=optimizers.Adam(), loss=heteroscedastic_loss, metrics=[mse_loss])
    assert len(model.layers[1].trainable_weights) == 3  # kernel, bias, and dropout prob
    assert len(model.losses) == 5, f'{len(model.losses)} is not 5'  # a loss for each Concrete Dropout layer

    return model
