from sklearn.preprocessing import MinMaxScaler


def scale(train_set, val_set, test_set=None, min_val=0, max_val=1, gap=0):
    scaler = MinMaxScaler((min_val+gap, max_val-gap))

    if train_set.ndim == 1:
        train_set = train_set.reshape(-1, 1)
        val_set = val_set.reshape(-1, 1)
        if test_set is not None:
            test_set = test_set.reshape(-1, 1)

    scaler.fit(train_set)
    train_set = scaler.transform(train_set)
    val_set = scaler.transform(val_set)
    if test_set is not None:
        test_set = scaler.transform(test_set)
        return scaler, train_set, val_set, test_set
    return scaler, train_set, val_set
