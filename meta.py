import autosklearn.metalearning
from autosklearn.constants import *
from autosklearn.metalearning.mismbo import suggest_via_metalearning
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.evaluation import ExecuteTaFuncWithQueue, WORST_POSSIBLE_RESULT
from autosklearn.util import get_logger
from autosklearn.metalearning.metalearning.meta_base import MetaBase
from autosklearn.metalearning.metafeatures.metafeatures import \
    calculate_all_metafeatures_with_labels, calculate_all_metafeatures_encoded_labels

from amb.data.profiler import DataProfiler
from amb.settings import settings, get_logger

logger = get_logger(__name__)


EXCLUDE_META_FEATURES_CLASSIFICATION = {
    'Landmark1NN',
    'LandmarkDecisionNodeLearner',
    'LandmarkDecisionTree',
    'LandmarkLDA',
    'LandmarkNaiveBayes',
    'PCAFractionOfComponentsFor95PercentVariance',
    'PCAKurtosisFirstPC',
    'PCASkewnessFirstPC',
    'PCA'
}

"""
Same as above except these are added:
    'NumberOfClasses',
    'ClassOccurences',
    'ClassProbabilityMin',
    'ClassProbabilityMax',
    'ClassProbabilityMean',
    'ClassProbabilitySTD',
    'ClassEntropy',
    'LandmarkRandomNodeLearner',
"""

EXCLUDE_META_FEATURES_REGRESSION = {
    'Landmark1NN',
    'LandmarkDecisionNodeLearner',
    'LandmarkDecisionTree',
    'LandmarkLDA',
    'LandmarkNaiveBayes',
    'PCAFractionOfComponentsFor95PercentVariance',
    'PCAKurtosisFirstPC',
    'PCASkewnessFirstPC',
    'NumberOfClasses',
    'ClassOccurences',
    'ClassProbabilityMin',
    'ClassProbabilityMax',
    'ClassProbabilityMean',
    'ClassProbabilitySTD',
    'ClassEntropy',
    'LandmarkRandomNodeLearner',
    'PCA',
}


def calculate_metafeatures(profile, basename, x_train, y_train):

    is_class = profile.has_categorical_target()
    pf = profile.get_raw_profile()
    categorical = pf.loc[~pf['drop'] & ~pf['target'] & ~(pf['col_type'] == 'datetime'), 'is_cat'].values

    if is_class:
        EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_CLASSIFICATION
    else:
        EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_REGRESSION

    logger.info('Start calculating metafeatures')
    result = calculate_all_metafeatures_with_labels(
        x_train, y_train, categorical=categorical,
        dataset_name=basename,
        dont_calculate=EXCLUDE_META_FEATURES, )
    for key in list(result.metafeature_values.keys()):
        if result.metafeature_values[key].type_ != 'METAFEATURE':
            del result.metafeature_values[key]
    return result

"""
def calculate_metafeatures_encoded(profile, basename, x_train, y_train):
    ## Assumes all are non-categorical

    is_class = profile.has_categorical_target()

    if is_class:
        EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_CLASSIFICATION
    else:
        EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_REGRESSION

    result = calculate_all_metafeatures_encoded_labels(
        x_train, y_train, categorical=[False] * x_train.shape[1],
        dataset_name=basename, dont_calculate=EXCLUDE_META_FEATURES)
    for key in list(result.metafeature_values.keys()):
        if result.metafeature_values[key].type_ != 'METAFEATURE':
            del result.metafeature_values[key]
    return result
"""

def get_metafeatures(df_train, target, name):
    # dont' one hot encode
    dp = DataProfiler(target=target, one_hot_encode=False)
    x_train, y_train = dp.fit_transform(df_train)

    is_class = dp.has_categorical_target()
    is_timeseries = dp.is_timeseries

    if is_class:
        # cast targets to int for classification
        y_train = y_train.astype(int)

    metafeats = calculate_metafeatures(dp, name, x_train.values, y_train.values)
    
    return metafeats, is_class, is_timeseries


if __name__ == '__main__':
    
    import os, sys
    sys.path.append(os.path.join(settings.ROOT_DIR, 'tests'))
    from datasets import Datasets, TestType
    from compare_models import datasets

    import pandas as pd
    from pprint import pprint

    dat = []
    bads = []

    for d in datasets: #[eval(sys.argv[1])]:
        
        df_train = d.train()
        target = d.meta().target
        name = d.meta().name

        metafeats, is_class, is_timeseries = get_metafeatures(df_train, target, name)

        is_class_true = d.meta().category == TestType.Classification

        if is_class != is_class_true:
            bads.append(d)
            continue

        arr = [(k, v.value) for k, v in metafeats.metafeature_values.items()]

        out = {}
        out['dataset_name'] = name
        out['is_class'] = is_class
        out['is_timeseries'] = is_timeseries
        out.update(dict(arr))

        pprint(out)

        dat.append(out)

    df = pd.DataFrame(dat)
    df.to_csv('/home/jgoode/amb-data/meta-feats.csv')

    print('bads sets: ', [x.meta().name for x in bads])

    #df2 = pd.read_csv('/home/jgoode/amb-data/meta-feats.csv', index_col='dataset_name').drop(['Unnamed: 0'], axis=1)