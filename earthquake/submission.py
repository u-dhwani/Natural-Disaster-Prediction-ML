from catboost import CatBoostRegressor
import pandas as pd

import config, utils


def get_features():
    data = utils.read_csv(config.path_to_train)
    features = [column for column in data.columns if column not in ['target', 'seg_id']]
    best_features = utils.feature_importance(data[features], data['target'], n_best=15, n_jobs=8)
    return best_features


def submit():
    """Make prediction and prepare results for submission.
    """

    features = get_features()

    train_set = pd.read_csv(config.path_to_train)
    test_set = pd.read_csv(config.path_to_test)

    model = CatBoostRegressor(verbose=False)
    model.fit(train_set[features], train_set['target'])

    results = pd.DataFrame()
    results['seg_id'] = test_set['seg_id']
    results['time_to_failure'] = model.predict(test_set[features])

    results.to_csv(config.path_to_results, index=False, float_format='%.5f')


if __name__ == '__main__':
    submit()
