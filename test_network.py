from data_gen import pendulum
import pickle
from models.mlp_tf import mlp, mlp_flipout
from models.cd import make_model
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model-type', default='de', type=str, help='model type')
parser.add_argument('--model-number', default='1', type=int, help='model number')
parser.add_argument('--t-spread-min', default=0.01, type=float, help='minimum T spread (rel noise on measurements)')
parser.add_argument('--t-spread-max', default=0.1, type=float, help='maximum T spread (rel noise on measurements)')
parser.add_argument('--ell-spread-min', default=0.02, type=float, help='minimum L spread (rel noise on measurements)')
parser.add_argument('--ell-spread-max', default=0.02, type=float, help='maximum L spread (rel noise on measurements)')
parser.add_argument('--ell-range-min', default=2, type=int, help='minimum 10*L')
parser.add_argument('--ell-range-max', default=8, type=int, help='maximum 10*L')
parser.add_argument('--g-range-min', default=5, type=int, help='minimum g')
parser.add_argument('--g-range-max', default=15, type=int, help='maximum g')
parser.add_argument('--n-test', default=10000, type=int, help='number of test examples')
parser.add_argument('--n-epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--data-dir', default='data/', type=str, help='data directory')

n_models = 10
n_neurons = 100


def main(model_type, model_number, t_spread_min, t_spread_max, ell_spread_min, ell_spread_max, ell_range_min,
         ell_range_max, g_range_min, g_range_max, n_test, n_epochs, data_dir):
    t_range_str = f'trange{int(100 * t_spread_min)}to{int(100 * t_spread_max)}'
    model_name = f'{model_type}_{t_range_str}_{n_epochs}ep'

    test_name = ''
    if ell_range_min != 2 or ell_range_max != 8:
        test_name += f'_ell{int(ell_range_min)}to{int(ell_range_max)}'
    if g_range_min != 5. or g_range_max != 15.:
        test_name += f'_g{int(g_range_min)}to{int(g_range_max)}'

    ell_range_min /= 10.
    ell_range_max /= 10.

    # Generate data
    feat_test, _, _, _ = pendulum(n=n_test, t_spread=[t_spread_min, t_spread_max],
                                  ell_range=[ell_range_min, ell_range_max], g_range=[g_range_min, g_range_max],
                                  ell_spread=[ell_spread_min, ell_spread_max], seed=666)

    with open(f'{data_dir}x_scaler_{t_range_str}.pkl', 'rb') as file_pi:
        x_scaler = pickle.load(file_pi)
    with open(f'{data_dir}y_scaler_{t_range_str}.pkl', 'rb') as file_pi:
        y_scaler = pickle.load(file_pi)

    feat_test = x_scaler.transform(feat_test)

    if model_type == 'de':
        models = [mlp(loss='nll') for _ in range(n_models)]
    elif model_type == 'cd':
        n_features = feat_test.shape[1]
        n_outputs = 1
        models = [make_model(n_features, n_outputs, n_neurons)]
    elif model_type == 'bnn':
        models = [mlp_flipout()]
    else:
        raise ValueError(f'Model type {model_type} not recognized!')

    for j, mod in enumerate(models):
        mod.load_weights(f'{data_dir}model_{model_name}_{str(model_number+j).zfill(3)}.h5')

    # make predictions
    if model_type == 'de':
        y_pred = []
        for model in models:
            y_pred.append(model(feat_test.astype('float32')))
    elif model_type == 'cd':
        y_pred = np.array([models[0].predict(feat_test) for _ in range(n_models)])
    elif model_type == 'bnn':
        y_pred = [models[0](feat_test.astype('float32')) for _ in range(n_models)]

    if model_type == 'de' or model_type == 'bnn':
        y_pred_val = [pred.loc.numpy() for pred in y_pred]
        y_pred_unc = [pred.scale.numpy() for pred in y_pred]
    elif model_type == 'cd':
        y_pred_val = y_pred[:, :, :1]
        y_pred_unc = np.sqrt(np.exp(y_pred[:, :, 1:]))

    y_pred_val_resc = [y_scaler.inverse_transform(y) for y in y_pred_val]
    y_pred_unc_resc = [y / y_scaler.scale_[0] for y in y_pred_unc]

    y_pred_val_resc = np.array(y_pred_val_resc).reshape((n_models, n_test))
    y_pred_unc_resc = np.array(y_pred_unc_resc).reshape((n_models, n_test))

    y_pred_mean = np.mean(y_pred_val_resc, axis=0)
    y_pred_ep_unc = np.std(y_pred_val_resc, axis=0)
    y_pred_al_unc = np.sqrt(np.mean(y_pred_unc_resc * y_pred_unc_resc, axis=0))
    y_pred_unc = np.sqrt(y_pred_al_unc ** 2 + y_pred_ep_unc ** 2)

    np.save(f'{data_dir}y_pred_test_{model_name}_{str(model_number).zfill(3)}{test_name}.npy', y_pred_mean)
    np.save(f'{data_dir}y_pred_test_alunc_{model_name}_{str(model_number).zfill(3)}{test_name}.npy', y_pred_al_unc)
    np.save(f'{data_dir}y_pred_test_epunc_{model_name}_{str(model_number).zfill(3)}{test_name}.npy', y_pred_ep_unc)
    np.save(f'{data_dir}y_pred_test_prunc_{model_name}_{str(model_number).zfill(3)}{test_name}.npy', y_pred_unc)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
