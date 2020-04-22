import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import tensorshape_util

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

n_train = 90000


class MeanMetricWrapper(tfk.metrics.Mean):
    # code by @mcourteaux from https://github.com/tensorflow/probability/issues/742#issuecomment-580433644
    def __init__(self, fn, name=None, dtype=None, **kwargs):
        super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        matches = self._fn(y_true, y_pred, **self._fn_kwargs)
        return super(MeanMetricWrapper, self).update_state(
            matches, sample_weight=sample_weight)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = K.eval(v) if is_tensor_or_variable(v) else v
        base_config = super(MeanMetricWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def scaled_kl_fn(a, b, _):
    """
    idea from
    https://github.com/google-research/google-research/blob/9645220c865ab5603b377e6a98265631ece61d44/uq_benchmark_2019/uq_utils.py
    https://arxiv.org/pdf/1906.02530.pdf
    :param a: distribution
    :param b: distribution
    :return: scaled kl divergence
    """
    return tfd.kl_divergence(a, b) / n_train


def mmd_from_dists(a, b, _):
    p = a.distribution
    q = b.distribution

    num_reduce_dims = (tensorshape_util.rank(a.event_shape) -
                       tensorshape_util.rank(p.event_shape))
    gamma_sq = 0.5
    reduce_dims = [-i - 1 for i in range(0, num_reduce_dims)]
    for i in reduce_dims:
        gamma_sq *= a.event_shape[i]

    sigma_p = tf.convert_to_tensor(tf.square(p.scale))
    sigma_q = tf.convert_to_tensor(tf.square(q.scale))
    scale_pp = gamma_sq + 2 * sigma_p
    scale_qq = gamma_sq + 2 * sigma_q
    scale_cr = gamma_sq + sigma_p + sigma_q

    return tf.reduce_sum(
        tf.math.sqrt(gamma_sq / scale_pp) + tf.math.sqrt(gamma_sq / scale_qq)
        - 2 * tf.math.sqrt(gamma_sq / scale_cr) * tf.math.exp(
            -0.5 * tf.math.squared_difference(p.loc, q.loc) / scale_cr),
        axis=reduce_dims)


def negloglik(y_data, rv_y):
    return -rv_y.log_prob(y_data)


def negloglik_met(y_true, y_pred):
    return tf.reduce_mean(-y_pred.log_prob(tf.cast(y_true, tf.float32)))


def mlp(hidden_dim=100, n_layers=3, n_inputs=13, dropout_rate=0, loss='mse'):
    input_data = tfkl.Input((n_inputs,))
    x = input_data
    for _ in range(n_layers):
        x = tfkl.Dense(hidden_dim, activation='relu')(x)
        if dropout_rate > 0:
            x = tfkl.Dropout(dropout_rate)(x)

    if loss == 'mse':
        x = tfkl.Dense(1)(x)
        model = tfk.Model(input_data, x)
        model.compile(loss='mean_squared_error', optimizer=tf.optimizers.Adam())
    elif loss == 'nll':
        x = tfkl.Dense(2)(x)
        x = tfpl.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                                                         scale=1e-3 + tf.math.softplus(t[..., 1:])))(x)
        model = tfk.Model(input_data, x)
        model.compile(optimizer=tf.optimizers.Adam(), loss=negloglik, metrics=['mse'])
    else:
        raise ValueError(f'Loss {loss} not implemented.')

    return model


def mlp_flipout(hidden_dim=100, n_layers=3, n_inputs=13, dropout_rate=0, kernel='kl'):
    input_img = tfkl.Input(n_inputs)
    x = input_img
    if kernel == 'kl':
        kernel_fn = scaled_kl_fn
    elif kernel == 'mmd':
        kernel_fn = mmd_from_dists
    else:
        raise ValueError(f'Kernel {kernel} not defined!')
    
    for _ in range(n_layers):
        x = tfpl.DenseFlipout(hidden_dim, activation='relu', kernel_divergence_fn=kernel_fn)(x)
        if dropout_rate > 0:
            x = tfkl.Dropout(dropout_rate)(x)
    x = tfpl.DenseFlipout(2, kernel_divergence_fn=kernel_fn)(x)
    x = tfpl.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                                                     scale=1e-3 + tf.math.softplus(t[..., 1:])))(x)
    model = tfk.Model(input_img, x)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), loss=negloglik,
                  metrics=['mse', MeanMetricWrapper(negloglik_met, name='nll')])

    return model
