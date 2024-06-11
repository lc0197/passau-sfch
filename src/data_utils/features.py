import itertools
import os
import pickle
from functools import partial
from glob import glob
from itertools import chain

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from global_vars import FEATURE_CACHE_DIR, COACHES, DATA_DIR, FEATURE_DIR, HUMOR


def load_feature_for_db_row(row, feature_df, reduction=None):
    '''
    One row in a database dataframe corresponds to a whole segment => load all features for this segment
    '''
    values = feature_df[(feature_df.segment_id==row.segment) & (feature_df.absolute_timestamp > row.start) &
                        (feature_df.absolute_timestamp <= row.end)].values[:,2:-1]
    if reduction is None:
        return values
    else:
        return reduction(values)

def load_feature_for_coach(db, coach, feature, task, cache_dir=FEATURE_CACHE_DIR, reduction=None, padding=0):
    '''
    Loads one specific feature for a coach

    :param db: dataframe
    :param coach: str
    :param feature: str
    :param cache_dir: optional, if None, no lookup. If given, but directory does not exist, it will be created.
    :return numpy array of shape (#coach segments, feature_dim)
    '''
    if not (cache_dir is None):
        cache_file = os.path.join(cache_dir, task, coach, f'{feature}.pickle')
        cache_file_lengths = os.path.join(cache_dir, task, coach, f'{feature}_lengths.pickle')
        if os.path.exists(cache_file) and os.path.exists(cache_file_lengths):
            features = pickle.load(open(cache_file, 'rb'))
            lengths = pickle.load(open(cache_file_lengths, 'rb'))
            return features, lengths
    else:
        cache_file = None
        cache_file_lengths = None

    csvs = sorted(glob(os.path.join(FEATURE_DIR, feature, coach, '*.csv')))
    feature_df = pd.concat([pd.read_csv(csv) for csv in csvs])
    feature_df['absolute_timestamp'] = feature_df['timestamp'].values + np.array(
        [int(s.split("_")[2]) + 500 for s in feature_df.segment_id.values])
    coach_db = db[db.coach == coach]

    # raw features if not segmentation
    raw_features = [load_feature_for_db_row(r, feature_df, reduction) for _, r in coach_db.iterrows()]
    if reduction is None:
        max_len = np.max([rf.shape[0] for rf in raw_features])
        lengths = [rf.shape[0] for rf in raw_features]
        raw_features = [np.pad(rf, ((0, max_len - rf.shape[0]), (0, 0)), mode='constant', constant_values=padding)
                        for rf in raw_features]
        # needed for the stacking
        raw_features = [np.expand_dims(rf, 0) for rf in raw_features]
    else:
        lengths = [1 for _ in raw_features]
    # else:
    #     # TODO build another function out of this...
    #     segments = sorted(list(set(feature_df.segment_id.values)))
    #     raw_features = []
    #     lengths = []
    #     for s in segments:
    #         #seg_df = coach_db[coach_db.segment_id==s]
    #         features = feature_df[feature_df.segment_id==s].values[:,2:-1]
    #         raw_features.append(np.expand_dims(features, 0))
    #         lengths.append(features.shape[1])
        print()
    features = np.concatenate(raw_features, axis=0)

    if not cache_file is None:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        pickle.dump(features, open(cache_file, 'wb+'))
        pickle.dump(lengths, open(cache_file_lengths, 'wb+'))
    return features, lengths


def load_feature_for_coach_segments(db, coach, feature, task, cache_dir=FEATURE_CACHE_DIR, reduction=None, padding=0):
    '''
    Loads one specific feature for a coach

    :param db: dataframe
    :param coach: str
    :param feature: str
    :param cache_dir: optional, if None, no lookup. If given, but directory does not exist, it will be created.
    :return numpy array of shape (#coach segments, feature_dim)
    '''
    if not (cache_dir is None):
        cache_file = os.path.join(cache_dir, task, coach, f'{feature}_seg.pickle')
        cache_file_lengths = os.path.join(cache_dir, task, coach, f'{feature}_seg_lengths.pickle')
        if os.path.exists(cache_file) and os.path.exists(cache_file_lengths):
            features = pickle.load(open(cache_file, 'rb'))
            lengths = pickle.load(open(cache_file_lengths, 'rb'))
            return features, lengths
    else:
        cache_file = None
        cache_file_lengths = None

    csvs = sorted(glob(os.path.join(FEATURE_DIR, feature, coach, '*.csv')))
    features = {}
    for csv in csvs:
        fdf = pd.read_csv(csv)
        feature_values = fdf.values[:,2:].astype(np.float32)
        if np.any(np.isnan(feature_values)):
            print(csv)
        feature_values = feature_values.astype(np.float32)
        features[os.path.basename(csv)[:-4]] = feature_values
    lengths = {seg_id: len(features[seg_id]) for seg_id in features.keys()}

    if not cache_file is None:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        pickle.dump(features, open(cache_file, 'wb+'))
        pickle.dump(lengths, open(cache_file_lengths, 'wb+'))
    return features, lengths

# Leave as is
def load_data_for_coaches(db, coaches, feature, target, normalize=False, scaler=None, reduction=None):
    '''
    Load features and labels for a set of coaches.

    :param db: dataframe
    :param coaches: list of coaches
    :param feature: string
    :param target: string (e.g. 'sentiment')
    :param normalize: if False, a given scaler will be ignored
    :param scaler: MinMaxScaler, ignored if normalize=False. If None and normalize True, it will be created
    :return (features (np array), labels (np array), scaler (MinMaxScaler or None), array indicating at which index the
        coaches' features start/end
    '''
    coach_data = [load_feature_for_coach(db, c, feature, target, reduction=reduction) for c in coaches]
    feature_arrs = [d[0] for d in coach_data]
    length_arrs = [d[1] for d in coach_data]
    coach_idxs = np.cumsum([0] + [len(a) for a in feature_arrs])
    features = np.concatenate(feature_arrs)
    lengths = np.concatenate(length_arrs)
    if normalize:
        feature_dim = features.shape[-1]
        unreduced = features.ndim==3
        if unreduced:
            features = features.reshape(features.shape[0], features.shape[1]*features.shape[2])

        if scaler is None:
            scaler = MinMaxScaler().fit(features)

        features = scaler.transform(features)
        if unreduced:
            features = features.reshape(features.shape[0], -1, feature_dim)

    labels = np.concatenate([db[db.coach == c][target].values for c in coaches])
    #print(features.shape)
    return features, lengths, labels, scaler, coach_idxs


def load_full_seg_data_for_coaches(db, coaches, feature, target, normalize=False, scaler=None, reduction=None):
    '''
    Load features and labels for a set of coaches.

    :param db: dataframe
    :param coaches: list of coaches
    :param feature: string
    :param target: string (e.g. 'sentiment')
    :param normalize: if False, a given scaler will be ignored
    :param scaler: MinMaxScaler, ignored if normalize=False. If None and normalize True, it will be created
    :return (features (np array), labels (np array), scaler (MinMaxScaler or None), array indicating at which index the
        coaches' features start/end
    '''
    coach_data = [load_feature_for_coach_segments(db, c, feature, target, reduction=reduction) for c in coaches]
    feature_dcts = [d[0] for d in coach_data]
    length_dicts = [d[1] for d in coach_data]
    coach_idxs = np.cumsum([0] + [len(d) for d in feature_dcts])

    if normalize:
        flat_features = list(itertools.chain.from_iterable([list(d.values()) for d in feature_dcts]))
        full_features = np.vstack(flat_features)

        if scaler is None:
            scaler = MinMaxScaler().fit(full_features)

        for i in range(len(feature_dcts)):
            for k,v in feature_dcts[i].items():
                feature_dcts[i][k] = scaler.transform(v)

    label_dcts = []
    for i,coach in enumerate(coaches):
        segments = list(feature_dcts[i].keys())
        coach_df = db[db.coach == coach]
        coach_labels = {seg_id: coach_df[coach_df.segment == seg_id].humor.values for seg_id in segments}
        for k,v in coach_labels.items():
            if len(v) == 0:
                print(k)
                exit(-1)
        label_dcts.append(coach_labels)
    #print(features.shape)
    # to arrays
    features = []
    lengths = []
    labels = []
    for i in range(len(feature_dcts)):
        for k in feature_dcts[i].keys():
            features.append(feature_dcts[i][k])
            lengths.append(length_dicts[i][k])
            labels.append(label_dcts[i][k])
    return features, lengths, labels, scaler, coach_idxs


def load_loo_ds(db, leave_out_coach, feature, target, normalize=False, reduction=None):
    assert leave_out_coach in COACHES
    train_coaches = sorted(list(set(COACHES) - {leave_out_coach}))
    train_X, train_lengths, train_y, scaler, coach_idxs = load_data_for_coaches(db, train_coaches, feature, target,
                                                                                normalize=normalize,
                                                                                reduction=reduction)
    test_X, test_lengths, test_y, _, _ = load_data_for_coaches(db, [leave_out_coach], feature, target,
                                                               normalize=normalize, scaler=scaler,
                                                               reduction=reduction)

    return (train_X, train_lengths, train_y, coach_idxs), (test_X, test_lengths, test_y)

def loo_mm_dcts(feature_dcts, label_dcts, leave_out):
    modalities = feature_dcts.keys()
    train_feats = {m:[feature_dcts[m][i] for i in range(len(feature_dcts[m])) if i!= leave_out]
                   for m in modalities}
    test_feats = {m: [feature_dcts[m][i] for i in range(len(feature_dcts[m])) if i == leave_out]
                   for m in modalities}
    train_labels = [label_dcts[i] for i in range(len(label_dcts)) if i != leave_out]
    test_labels = [label_dcts[i] for i in range(len(label_dcts)) if i == leave_out]

    return (train_feats, train_labels), (test_feats, test_labels)


def loo_from_whole_ds(X, lengths, y, coach_idxs, leave_out):
    leave_out_idxs = np.arange(coach_idxs[leave_out], coach_idxs[leave_out+1], 1)

    train = np.ones((coach_idxs[-1],)).astype(np.int8)
    train[leave_out_idxs] = 0
    test = np.zeros((coach_idxs[-1],)).astype(np.int8)
    test[leave_out_idxs] = 1

    train = train.astype(np.bool8)
    test = test.astype(np.bool8)

    train_X = X[train]
    train_lengths = lengths[train]
    train_y = y[train]

    test_X = X[test]
    test_lengths = lengths[test]
    test_y = y[test]

    left_out_length = len(leave_out_idxs)
    subtraction = np.zeros_like(coach_idxs)
    subtraction[leave_out+1:] = left_out_length
    idxs_after_leave_out = np.delete((coach_idxs - subtraction),leave_out)

    return (train_X, train_lengths, train_y), idxs_after_leave_out, (test_X, test_lengths, test_y)


def loo_from_whole_seg_ds(X, lengths, y, coach_idxs, leave_out):
    leave_out_idxs = np.arange(coach_idxs[leave_out], coach_idxs[leave_out+1], 1)

    train = np.ones((coach_idxs[-1],)).astype(np.int8)
    train[leave_out_idxs] = 0
    test = np.zeros((coach_idxs[-1],)).astype(np.int8)
    test[leave_out_idxs] = 1

    train = train.astype(np.bool8)
    test = test.astype(np.bool8)

    train_X = X[train]
    train_lengths = lengths[train]
    train_y = y[train]

    test_X = X[test]
    test_lengths = lengths[test]
    test_y = y[test]

    left_out_length = len(leave_out_idxs)
    subtraction = np.zeros_like(coach_idxs)
    subtraction[leave_out+1:] = left_out_length
    idxs_after_leave_out = np.delete((coach_idxs - subtraction),leave_out)

    return (train_X, train_lengths, train_y), idxs_after_leave_out, (test_X, test_lengths, test_y)


def load_whole_ds(db, feature, target, normalize=False, reduction=None, segments=False):
    train_coaches = COACHES
    if not segments:
        train_X, train_lengths, train_y, scaler, coach_idxs = load_data_for_coaches(db, train_coaches, feature, target,
                                                                 normalize=normalize,
                                                                 reduction=reduction)
    else:
        train_X, train_lengths, train_y, scaler, coach_idxs = load_full_seg_data_for_coaches(db, train_coaches, feature, target,
                                                                                    normalize=normalize,
                                                                                    reduction=reduction)
    #coach_idxs = np.hstack([np.array((coach_idxs[i + 1] - coach_idxs[i]) * [i]) for i in range(len(train_coaches))])
    return train_X, train_lengths, train_y, coach_idxs


def load_multimodal_segm_ds(db, feature_v, feature_a, feature_t, normalize_v, normalize_a, normalize_t):
    feature_dcts = {}
    length_dcts = {}
    scalers = []

    modalities = ['v', 'a', 't']
    for i,(feature,normalize) in enumerate(list(zip([feature_v, feature_a, feature_t],
                                                    [normalize_v, normalize_a, normalize_t]))):
        modality = modalities[i]
        coach_data = [load_feature_for_coach_segments(db, c, feature, HUMOR, reduction=None) for c in COACHES]
        feature_dcts[modality] = [d[0] for d in coach_data]

        if normalize:
            # TODO copy from below
            flat_features = list(itertools.chain.from_iterable([list(d.values()) for d in feature_dcts[modality]]))
            full_features = np.vstack(flat_features)
            scaler = MinMaxScaler().fit(full_features)
            for i in range(len(feature_dcts[modality])):
                for k, v in feature_dcts[modality][i].items():
                    feature_dcts[modality][i][k] = scaler.transform(v)
            scalers.append(scaler)
        else:
            scalers.append(None)
        length_dcts[modality] = [d[1] for d in coach_data]

    # TODO check for missing, fix s.t. all modalities have same segments and lengths
    all_segs = [set() for _ in range(len(feature_dcts['a']))]
    for m in modalities:
        for i,d in enumerate(feature_dcts[m]):
            keys = set(d.keys())
            all_segs[i].update(keys)
    # which segments are missing?
    for m in modalities:
        for i,d in enumerate(feature_dcts[m]):
            expected_segs = all_segs[i]
            missing_segs = expected_segs - set(d.keys())
            if len(missing_segs) > 0:
                present_segs = list(expected_segs - missing_segs)
                feature_size = d[present_segs[0]].shape[1]
                for s in missing_segs:
                    seq_length = None
                    for m2 in set(modalities) - {m}:
                        if s in feature_dcts[m2][i]:
                            seq_length = feature_dcts[m2][i][s].shape[0]
                            break
                    if seq_length is None:
                        print(f'Could not reconstruct missing segment {s} for modality {m}')
                    dummy = np.zeros((seq_length, feature_size))
                    feature_dcts[m][i][s] = dummy
            print()

    # TODO do we even need length_dcts???
    #length_dcts = None
    #coach_idxs = np.cumsum([0] + [len(d) for d in feature_dcts])



    # TODO check for segments
    label_dcts = []
    for i, coach in enumerate(COACHES):
        segments = list(feature_dcts['v'][i].keys())
        coach_df = db[db.coach == coach]
        coach_labels = {seg_id: coach_df[coach_df.segment == seg_id].humor.values for seg_id in segments}
        label_dcts.append(coach_labels)

    return feature_dcts, label_dcts#, scaler


def get_feature_dim(feature):
    # load some csv
    csvs = glob(os.path.join(FEATURE_DIR, feature, 'baum', '*.csv'))
    feature_df = pd.read_csv(csvs[0])
    return feature_df.shape[1] - 3


def idx_splitting(idxs):
    """
    Auxiliary method for leave-one-out strategy. This is an iterable that can be used as 'cv' in sklearn's GridSearchCV

    :param idxs: array containing (num_coaches_in_partition)+1 increasing indices (coach_idxs as returned by load_loo_ds)
    :return iterable of tuples (train_idxs, test_idxs)
    """
    for i in range(len(idxs)-1):
        train = []
        test = []
        for k in range(i):
            train.extend(list(range(idxs[k], idxs[k+1])))
        test.extend(list(range(idxs[i], idxs[i+1])))
        for k in range(i+1, len(idxs)-1):
            train.extend(list(range(idxs[k], idxs[k+1])))
        yield (train, test)


if __name__ == '__main__':
    db = pd.read_csv(os.path.join(DATA_DIR, 'gold_standards', '10_aao_mean_ccc', 'gs.csv'))
    db['coach'] = [f.split("_")[0] for f in db.file.values]
    db['start'] = db['start'].values - 500
    feature = 'egemaps'
    target = 'sentiment'
    margin = 0.05
    reduction = lambda x: np.mean(x, axis=1)
    # load_whole_ds_three_class(db, feature, target, margin=margin, normalize=True, reduction=reduction)

    x = load_data_for_coaches(db, ['baum', 'breitenreiter'], 'egemaps', 'sentiment', reduction=None)
    print(x)