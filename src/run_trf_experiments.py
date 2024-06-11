import itertools
import os
import sys
from argparse import ArgumentParser, Namespace
from collections import Counter
from itertools import product
from time import ctime, time

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils.features import load_whole_ds, loo_from_whole_ds, load_multimodal_segm_ds, loo_mm_dcts
from data_utils.labels import load_task_data, stratify_mm
from data_utils.log import unflatten_dict, summarize_results
from global_vars import HUMOR, RESULT_DIR, COACHES, device, PREDICTIONS_DIR, CHECKPOINT_DIR
from models.rnn import GRUClassifier
import numpy as np
import json

import pandas as pd

from models.transformers import EMB_TYPES, MODEL_TYPES, get_model_class, V_FOCUS
from train_eval.evaluation import evaluate_cont

from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--summarization_keys', required=False, default=['test.auc.mean'], nargs='+',
                        help='Create excel files for the corresponding results')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--save_checkpoints', action='store_true')

    ##### FEATURE PARAMETERS #######
    parser.add_argument('--features_v', default='farl')
    parser.add_argument('--features_a', default='wav2vec')
    parser.add_argument('--features_t', default='electra-4-sentence-level')
    parser.add_argument('--normalize', nargs=3, default=None, type=int, choices=[0,1],
                        help='Specify which features should be normalized. 1 for normalization, 0 for no normalization. '
                             'Must give three numbers.')

    ##### HYPERPARAMETERS #######
    # transformer parameters
    parser.add_argument('--trf_num_heads', type=int, nargs='+', default=[4])
    parser.add_argument('--trf_num_v_layers', type=int, nargs='+', default=[1])
    parser.add_argument('--trf_num_at_layers', type=int, nargs='+', default=[1])
    parser.add_argument('--trf_num_mm_layers', type=int, nargs='+', default=[1])
    parser.add_argument('--trf_pos_emb', type=str, nargs='+', choices=EMB_TYPES, default=EMB_TYPES)
    parser.add_argument('--trf_model_dim', type=int, nargs='+', default=[64])
    parser.add_argument('--model_type', type=str, choices=MODEL_TYPES, default=V_FOCUS)
    parser.add_argument('--context_window', type=str, required=False, nargs='+', help='Integer or "None"')
    # training parameters
    parser.add_argument('--lr', type=float, nargs='+', default=[0.01, 0.001, 0.005, 0.0001, 0.0005])
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--seed', type=int, default = 101)
    parser.add_argument('--rnn_dropout', type=float, nargs='+', default=[0.])
    parser.add_argument('--regularization', nargs='+', type=float, default=[0.])
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--devtest_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max_training_length', type=int, default=128)


    args = parser.parse_args()
    if args.context_window is None:
        args.context_window = [None]
    else:
        cw = [None if c=='None' else int(c) for c in args.context_window]
        args.context_window = cw

    if args.normalize is None:
        args.normalize = [False] * 3
    else:
        args.normalize = [n > 0 for n in args.normalize]

    if args.name is None:
        args.name = 'trf_humor'
    args.name += str(ctime(time())).replace(":","-").replace(" ", "_")

    return args


# TODO extract? - should work
class MMDataset(Dataset):

    def __init__(self, feature_dcts, label_dcts):
        # unpack dicts
        all_segments = sorted(list((
            itertools.chain.from_iterable([
                list(d.keys()) for d in feature_dcts['v']
            ])
        )))
        self.Xs = []
        for m in ['v', 'a', 't']:
            flat_dct = {}
            m_features = []
            for d in feature_dcts[m]:
                flat_dct.update(d)
            for segment in all_segments:
                m_features.append(flat_dct[segment])
            self.Xs.append(m_features)
        flat_labels = {}
        self.labels = []
        self.coaches = []
        for d in label_dcts:
            flat_labels.update(d)
            coach = list(d.keys())[0].split("_")[0]
            self.coaches.extend([coach] * len(d))
        # min_length = np.min([a.shape[0] for a in flat_labels.values()])
        print([s for s in flat_labels.keys() if flat_labels[s].shape[0] < 4])
        #print()
        for segment in all_segments:
            self.labels.append(flat_labels[segment])
        # coaches to numbers
        coach2id = {c:i for i,c in enumerate(sorted(list(set(self.coaches))))}
        self.coaches = [coach2id[c] for c in self.coaches]


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return (self.Xs[0][item], self.Xs[1][item], self.Xs[2][item]), self.coaches[item], self.labels[item]

    def get_dims(self):
        return self.Xs[0][0].shape[1], self.Xs[1][0].shape[1], self.Xs[2][0].shape[1]

    def get_flat_labels(self):
        return np.array(list(itertools.chain.from_iterable(self.labels)))

def create_mask(lens, pad_to=None):
    max_len = np.max(lens) if pad_to is None else max(np.max(lens), pad_to)
    mask = np.zeros((len(lens), max_len))
    for i,l in enumerate(lens):
        mask[i,:l] = 1.
    return mask

def mm_collate_fn(batch, ignore_index=-100):
    #batch_size = len(batch)
    # TODO pad to multiple of 4?
    X_vs = [b[0][0] for b in batch]
    lens = [X_v.shape[0] for X_v in X_vs]
    X_as = [b[0][1] for b in batch]
    X_ts = [b[0][2] for b in batch]
    #assert all(X.shape[0]==X_vs.shape[0] for X in [X_as, X_ts])
    # always pad to multiple of 2 or 4, depending on label length
    labels = [np.expand_dims(b[2], 0) for b in batch]
    max_label_len = np.max([l.shape[1] for l in labels])

    max_x_len = np.max(lens)
    pad_to = max_x_len
    for i in range(5):
        if np.floor((max_x_len + i - 4) / 2 + 1).astype(np.int32) == max_label_len:
            pad_to = max_x_len + i
            break
    #print(max_x_len, max_label_len, pad_to)
    masks = create_mask(lens, pad_to=pad_to)

    X_padded_v = [np.pad(x, ((0, pad_to - x.shape[0]), (0,0))) for x in X_vs]
    X_vs = np.vstack([np.expand_dims(x, 0) for x in X_padded_v])
    X_padded_a = [np.pad(x, ((0, pad_to - x.shape[0]), (0, 0))) for x in X_as]
    X_as = np.vstack([np.expand_dims(x, 0) for x in X_padded_a])
    X_padded_t = [np.pad(x, ((0, pad_to - x.shape[0]), (0, 0))) for x in X_ts]
    X_ts = np.vstack([np.expand_dims(x, 0) for x in X_padded_t])

    labels = [np.pad(l, ((0,0), (0, max_label_len - l.shape[1])), constant_values=ignore_index) for l in labels]
    labels = np.vstack(labels)

    coaches = [b[1] for b in batch]
    return (torch.FloatTensor(X_vs), torch.FloatTensor(X_as), torch.FloatTensor(X_ts)), torch.LongTensor(coaches), torch.tensor(masks), torch.tensor(labels)


def init_model(params:Namespace, seed:int, num_classes:int=2):
    torch.manual_seed(seed)
    model = GRUClassifier(params=params, num_classes=num_classes)
    model.to(device)
    return model


def init_weights(y:np.ndarray):
    '''

    Balances the classes
    :param y: shape (num_examples,)
    :return:
    '''
    counts = {cls:cnt for (cls,cnt) in Counter(y).most_common()}
    classes = sorted(list(np.unique(y)))
    return torch.tensor([y.shape[0]/counts[classes[i]] for i in range(len(classes))])



def training_epoch(model, optimizer, train_loader, loss_fn, writer, prev_steps):
    model.train()
    step_counter = prev_steps

    for batch in tqdm(train_loader):
        step_counter += 1
        optimizer.zero_grad()
        Xs, coaches, masks, y = batch

        logits = model(Xs[0].to(device), Xs[1].to(device), Xs[2].to(device), masks.to(device))
        #print(logits.shape)
        logits = logits.squeeze(-1)
        #print(logits.shape)
        #print(y.shape)
        valid_idxs = y > -100

        y = y[valid_idxs]
        logits = logits[valid_idxs]
        loss = loss_fn(logits, y.float().to(device))
        loss_value = loss.detach().cpu().numpy()
        loss.backward()
        if not writer is None:
            writer.add_scalar('train_loss', loss_value, step_counter)
            # if not logits_aux is None:
            #     writer.add_scalar('train_w_aux', loss.detach().cpu().numpy(), step_counter)
        optimizer.step()
    return model, step_counter

def extract_logits_and_gs(model, val_loader:DataLoader):
    '''

    Extract both model predictions (logits) and gold standard values, given a DataLoader
    :param model: model
    :param val_loader: loader
    :return: tensor of logits, numpy array of gold standard values
    '''
    model.eval()

    all_logits = []
    all_ys = []

    with torch.no_grad():
        for batch in val_loader:
            Xs, _, masks, y = batch
            logits = model(Xs[0].to(device), Xs[1].to(device), Xs[2].to(device), masks.to(device)).squeeze(-1)
            valid_idxs = y > -100
            logits = logits[valid_idxs]
            y = y[valid_idxs]
            all_logits.append(logits.flatten().detach())
            all_ys.append(y.flatten().detach().cpu().numpy())

    all_logits = torch.cat(all_logits)
    all_ys = np.concatenate(all_ys)

    return all_logits, all_ys


def val_score(model, val_loader:DataLoader) -> float:
    '''

    Calculates ROC-AUC for the predictions of a model. Used to get a validation score
    :param model: model to evaluate
    :param val_loader: evaluation data
    :return: ROC-AUC score
    '''
    model.eval()
    sigmoid = nn.Sigmoid()

    all_logits, all_ys = extract_logits_and_gs(model, val_loader)

    predictions = sigmoid(all_logits)

    predictions = predictions.squeeze().cpu().numpy()

    score = roc_auc_score(all_ys.astype(np.int32), predictions)
    return score


def test_evaluation(model, test_loader:DataLoader) -> dict:
    '''

    Calculates test metrics for the predictions of a model. Currently identical to evaluation on the validation set
    (ROC-AUC), but allows for computing several metrics.
    :param model: model to evaluate
    :param test_loader: test data
    :return: dictionary of metrics and their value
    '''
    model.eval()
    sigmoid = nn.Sigmoid()
    all_logits, all_ys = extract_logits_and_gs(model, test_loader)

    cont_predictions = sigmoid(all_logits).detach().cpu().numpy()

    return evaluate_cont(all_ys, cont_predictions)


def extract_predictions(model, test_loader:DataLoader) -> dict:
    '''

    Extracts dictionary of actual predictions (sigmoid applied to logits)
    :param model: model to extract predictions of
    :param test_loader: DataLoader
    :return: prediction dict (can be used to immediately create a DataFrame)
    '''
    model.eval()
    sigmoid = nn.Sigmoid()
    all_logits, all_ys = extract_logits_and_gs(model, test_loader)

    predictions = sigmoid(all_logits).detach().cpu().numpy()

    if len(predictions.shape) == 1:
        predictions = np.expand_dims(predictions, -1)
    return {f'pred_{i}':predictions[:,i] for i in range(predictions.shape[1])}


def training(model, optimizer:torch.optim.Optimizer, loss_fn, epochs, patience, train_loader, val_loader,
            writer: SummaryWriter):
    '''

    Training routine
    :return: (best validation score, updated model)
    '''
    best_quality = -1234.
    patience_counter = 0
    best_state_dict = None

    step_counter=0
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}')
        model, step_counter = training_epoch(model, optimizer, train_loader=train_loader, loss_fn=loss_fn,
                                             writer=writer, prev_steps=step_counter)
        epoch_result = val_score(model, val_loader)
        if not (writer is None):
            writer.add_scalar('auc', epoch_result, epoch)

        if epoch_result > best_quality:
            best_quality = epoch_result
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter > patience:
                break

    model.load_state_dict(best_state_dict)
    return best_quality, model


if __name__ == '__main__':
    args = parse_args()
    db, label_mapping, target_col = load_task_data(task=HUMOR)

    configurations = [
        Namespace(**{'trf_num_heads': c[0], 'trf_num_v_layers': c[1], 'trf_num_at_layers': c[2], 'lr': c[3],
                     'regularization': c[4], 'trf_pos_emb': c[5], 'trf_model_dim': c[6], 'trf_num_mm_layers': c[7],
                      'context_window': c[8]}) for c in
        product(args.trf_num_heads, args.trf_num_v_layers, args.trf_num_at_layers, args.lr, args.regularization,
                args.trf_pos_emb, args.trf_model_dim, args.trf_num_mm_layers, args.context_window)]

    experiment_str = os.path.join(HUMOR, 'trf', "_".join([args.features_v, args.features_a, args.features_t]))

    res_dir = os.path.join(RESULT_DIR, experiment_str, args.name)
    os.makedirs(res_dir, exist_ok=True)

    result_json = os.path.join(res_dir, 'all_results.json')
    coach_excels = [os.path.join(res_dir, f'{key}_coaches.xlsx') for key in args.summarization_keys]
    feature_excels = [os.path.join(res_dir, f'{key}_features.xlsx') for key in args.summarization_keys]

    pred_dir = os.path.join(PREDICTIONS_DIR, experiment_str, args.name)

    sorted_label_names = [label_mapping[k] for k in sorted(list(label_mapping.keys()))]

    scoring = roc_auc_score

    res_dict = {'config': {'params': {k:vars(args)[k] for k in ['trf_num_heads', 'trf_num_v_layers', 'trf_num_at_layers',
                                                                'lr', 'regularization', 'trf_pos_emb', 'trf_model_dim',
                                                                'trf_num_mm_layers', 'context_window']},
                           'target': target_col,
                           'cli_args': vars(args)},
                'results': {}}

    feature_dcts, label_dcts = load_multimodal_segm_ds(db,
                                                     feature_v=args.features_v,
                                                     feature_a=args.features_a,
                                                     feature_t=args.features_t,
                                                     normalize_v=args.normalize[0],
                                                     normalize_a=args.normalize[1],
                                                     normalize_t=args.normalize[2])

    num_classes = 2

    best_config = configurations[0]
    best_score = 0

    if len(configurations) > 1:
        csv = os.path.join(res_dir, 'results.csv')
        print(f"Starting gridsearch for {len(configurations)} configurations")
        for n,config in tqdm(list(enumerate(configurations))):
                #print(f'Hyperparameter Search: Training {n+1} of {len(configurations)} configurations')
            seed_scores = []
            for seed in range(args.seed, args.seed+args.num_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                    # folds
                coach_scores = []
                for i in tqdm(range(len(COACHES))):
                    tb_dir = os.path.join(res_dir, 'tb', 'hp_search', f'{n}_{seed}_{i}')
                    tb = SummaryWriter(tb_dir)
                    (train_features, train_labels), (val_features, val_labels) = loo_mm_dcts(feature_dcts, label_dcts, leave_out=i)

                    train_loader = DataLoader(MMDataset(train_features, train_labels), batch_size=args.train_batch_size,
                                                  shuffle=True, collate_fn=mm_collate_fn)
                    val_loader = DataLoader(MMDataset(val_features, val_labels), batch_size=args.devtest_batch_size,
                                                shuffle=False, collate_fn=mm_collate_fn)
                    v_dim, a_dim, t_dim = train_loader.dataset.get_dims()
                    config.v_dim = v_dim
                    config.a_dim = a_dim
                    config.t_dim = t_dim
                    config.max_length = 1024


                    model = get_model_class(args.model_type)(config)
                    model.to(device)

                    optimizer = AdamW(lr = config.lr, params=model.parameters(), weight_decay=config.regularization)

                    flat_labels = train_loader.dataset.get_flat_labels()
                    weights = init_weights(flat_labels).float()

                    # make sure negative class has weight 1
                    weights = weights / weights[0]
                    loss_fn = BCEWithLogitsLoss(pos_weight=weights[1].to(device), reduction='mean')

                    coach_scores.append(training(model, optimizer, loss_fn, epochs=args.epochs, patience=args.patience,
                                                     train_loader=train_loader, val_loader=val_loader, writer=tb)[0])
                    tb.close()
                seed_scores.append(np.mean(coach_scores))

            config_score = np.mean(seed_scores)
            print(f'Finished training for config {config}')
            print(f'Score: {np.mean(seed_scores)}')
            if config_score > best_score:
                best_score = config_score
                best_config = config
            if os.path.exists(csv):
                res_df = pd.read_csv(csv)
            else:
                res_df = pd.DataFrame(columns=list(vars(config).keys()) + ['AUC'])
            new_row = vars(config)
            new_row['AUC'] = config_score
            res_df = res_df.append(new_row, ignore_index=True)
            res_df.to_csv(csv, index=False)
        config = best_config

    else:
        print('Only one config given, no hyperparameter search.')
        config = configurations[0]
    # load best config and test
    print()
    if len(configurations) > 1:
        print(f'{ctime(time())}: finished gridsearch - starting eval')

    coach_results = {coach:{'test':{}} for coach in COACHES}

    if args.save_predictions:
        feature_pred_dir = os.path.join(pred_dir, "_".join([args.features_v, args.features_a, args.features_t]))
        os.makedirs(feature_pred_dir, exist_ok=True)

    if args.save_checkpoints:
        cp_dir = os.path.join(CHECKPOINT_DIR, experiment_str, args.name)
        os.makedirs(cp_dir, exist_ok=True)

    print(f'Training {"best" if len(configurations) > 1 else "given"} configuration (leave-one-out scenario)')
    for i,coach in tqdm(list(enumerate(COACHES))):

        (train_features, train_labels), (test_features, test_labels) = loo_mm_dcts(feature_dcts, label_dcts, leave_out=i)

        test_loader = DataLoader(MMDataset(test_features, test_labels), batch_size=args.devtest_batch_size,
                                  shuffle=False, collate_fn=mm_collate_fn)

        seed_scores = []

        # which of the n_seeds models for prediction?
        best_test_seed_quality = -1234
        best_test_seed = None
        best_state_dict = None
        test_seed_predictions = {s:None for s in range(args.seed, args.seed+args.num_seeds)}

        for seed in range(args.seed, args.seed+args.num_seeds):
            tb_dir = os.path.join(res_dir, 'tb', 'best', f'best_config_{i}_{seed}')
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(tb_dir)
            torch.manual_seed(seed)
            np.random.seed(seed)
            (train_features, train_labels), (val_features, val_labels) = stratify_mm(train_features, train_labels, seed=seed)

            train_loader = DataLoader(MMDataset(train_features, train_labels), batch_size=args.train_batch_size,
                                      shuffle=True, collate_fn=mm_collate_fn)
            val_loader = DataLoader(MMDataset(val_features, val_labels), batch_size=args.devtest_batch_size,
                                    shuffle=False, collate_fn=mm_collate_fn)

            #config = best_config
            v_dim, a_dim, t_dim = train_loader.dataset.get_dims()
            config.v_dim = v_dim
            config.a_dim = a_dim
            config.t_dim = t_dim
            config.max_length = 1024  # args.max_training_length

            model = get_model_class(args.model_type)(config)
            num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
            print(f'Number of parameters: {num_params}')
            model.to(device)

            optimizer = AdamW(lr=config.lr, params=model.parameters(), weight_decay=config.regularization)

            flat_labels = train_loader.dataset.get_flat_labels()
            weights = init_weights(flat_labels).float()

            # make sure negative class has weight 1
            weights = weights / weights[0]
            loss_fn = BCEWithLogitsLoss(pos_weight=weights[1], reduction='mean')

            q, model = training(model, optimizer, loss_fn, epochs=args.epochs, patience=args.patience,
                                             train_loader=train_loader, val_loader=val_loader, writer=writer)
            seed_scores.append(test_evaluation(model, test_loader))
            if q > best_test_seed_quality:
                best_test_seed_quality = q
                best_test_seed = seed
                best_state_dict = model.state_dict()
            writer.close()

        coach_results[coach]['best_val'] = best_test_seed_quality
        sss_as_lists = {k:[s[k] for s in seed_scores] for k in seed_scores[0].keys()}
        for k in seed_scores[0].keys():
            coach_results[coach]['test'].update({f'{k}.mean': np.mean(sss_as_lists[k]),
                                             f'{k}.std':np.std(sss_as_lists[k])})

        res_dict['results']["_".join([args.features_v, args.features_a, args.features_t])] = {'results': unflatten_dict(coach_results)}

        # in case predictions should be saved
        if args.save_predictions:
            prediction_csv = os.path.join(feature_pred_dir, f'{coach}.csv')
            model.load_state_dict(best_state_dict)
            test_predictions = extract_predictions(model, test_loader)
            pd.DataFrame(test_predictions).to_csv(prediction_csv, index=False)

        if args.save_checkpoints:
            cp_file = os.path.join(cp_dir, f'{coach}_left_out.pt')
            torch.save(best_state_dict, cp_file)

    for i,summarization_key in enumerate(args.summarization_keys):
        summarize_results(res_dict, coach_target_file=coach_excels[i], feature_target_file=feature_excels[i],
                          key=summarization_key)

    res_dict['results'] = res_dict['results']["_".join([args.features_v, args.features_a, args.features_t])]['results']
    res_dict['best_config'] = vars(best_config)
    json.dump(res_dict, open(result_json, 'w+'))