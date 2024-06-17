from myfm import MyFMRegressor
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_squared_error
from math import sqrt

def train(dataset, iterator, rank=4, n_iter=200, n_kept_samples=200):
    print('Training Bayesian Factorization Machine...')
    FEATURE_COLUMNS = ['user', 'movie']
    ohe = OneHotEncoder(handle_unknown='ignore')
    X = ohe.fit_transform(dataset[FEATURE_COLUMNS])
    y = dataset['rating'].values

    rmse_values = []

    for train_indices, test_indices in iterator:
        print(f'Training on {len(train_indices)} samples, testing on {len(test_indices)} samples...')
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        fm = MyFMRegressor(rank=rank, random_seed=42)
        fm.fit(X_train, y_train, n_iter=n_iter, n_kept_samples=n_kept_samples)

        y_pred = fm.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        rmse_values.append(rmse)
        print(f'RMSE: {rmse}')

    average_rmse = sum(rmse_values) / len(rmse_values)
    print(f'Average RMSE: {average_rmse}')
