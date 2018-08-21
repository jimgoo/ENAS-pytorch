# -*- encoding: utf-8 -*-
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import numpy as np
import autosklearn.regression

import sys, os
from amb.settings import settings
sys.path.append(os.path.join(settings.ROOT_DIR, 'tests'))
from datasets import Datasets
from amb.data.profiler import DataProfiler


if __name__ == '__main__':
    # X, y = sklearn.datasets.load_boston(return_X_y=True)

    d = Datasets.Basic.cancer
    df_train = d.train()
    df_test = d.test()
    target = d.meta().target
    
    dp = DataProfiler(target=target, one_hot_encode=False)
    x_train, y_train = dp.fit_transform(df_train)
    x_test, y_test = dp.transform(df_test)

    y_train = y_train.dot(np.arange(y_train.shape[1])).astype(int)
    y_test = y_test.dot(np.arange(y_test.shape[1])).astype(int)
    
    pf = dp.get_raw_profile()
    import ipdb; ipdb.set_trace()

    is_cat = pf.loc[~pf['drop'] & ~pf['target'] & ~(pf['col_type'] == 'datetime'), 'is_cat'].values
    feature_types = ['categorical' if x else 'numerical' for x in is_cat]

    import meta
    # ipdb.set_trace()
    r1 = meta.calculate_metafeatures(dp, d.meta().name, x_train.values, y_train.values)
    r2 = meta.calculate_metafeatures_encoded(dp, d.meta().name, x_train.values, y_train.values)

    # # X_train, X_test, y_train, y_test = \
    # #     sklearn.model_selection.train_test_split(X, y, random_state=1)

    # automl = autosklearn.regression.AutoSklearnRegressor(
    #     time_left_for_this_task=120,
    #     per_run_time_limit=30,
    #     tmp_folder='/tmp/autosklearn_regression_example_tmp',
    #     output_folder='/tmp/autosklearn_regression_example_out',
    # )
    # automl.fit(X_train, y_train, dataset_name='boston',
    #            feat_type=feature_types)

    # print(automl.show_models())
    # predictions = automl.predict(X_test)
    # print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))
    # print(automl.sprint_statistics())
