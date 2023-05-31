import ot
import copy
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, List, Union
from sklearn import tree
from itertools import product
from collections import defaultdict
from scipy.stats import entropy
from sklearn.tree import DecisionTreeClassifier, BaseDecisionTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from mollifiers.booster import generate_adversarial_labels
from mollifiers.my_types import ContinuousDataInfo, DiscreteDataInfo
from mollifiers.hypothesis import Hypothesis, TorchHypothesis
from mollifiers.mollifier import DiscreteFairDensity

def representation_rate(samples: np.ndarray, info: Union[ContinuousDataInfo, DiscreteDataInfo]) -> float:
    sens_col = np.array(samples[:, info.sensitive_columns])

    s_rates = []
    for s in info.sensitive_domain:
        s_rates.append(float(np.sum(sens_col == s)))

    rr_values = []
    for i in range(len(s_rates)):
        for j in range(i+1, len(s_rates)):
            if s_rates[i] == 0 and s_rates[j] == 0:
                cur_rr = 1
            elif s_rates[i] == 0:
                cur_rr = s_rates[i] / s_rates[j]
            elif s_rates[j] == 0:
                cur_rr = s_rates[j] / s_rates[i]
            else:
                cur_rr = min(s_rates[i] / s_rates[j], s_rates[j] / s_rates[i])

            rr_values.append(cur_rr)
    
    return min(rr_values)

def statistical_rate(samples: np.ndarray, info: Union[ContinuousDataInfo, DiscreteDataInfo], delta = 0) -> float:
    samples = np.array(samples)
    
    sens_cols = samples[:, info.sensitive_columns].squeeze()
    label_cols = samples[:, info.label_columns].squeeze()

    if info.positive_label[0] == 0:
        label_cols = 1 - label_cols

    y1s1 = sum(label_cols * sens_cols)
    y1s0 = sum(label_cols * (1 - sens_cols))

    if delta == 0:
        s1 = sum(sens_cols) + delta
        s0 = len(samples) - s1 + delta
    else:
        old_s1 = sum(sens_cols) + delta
        old_s0 = len(samples) - old_s1 + delta

        s1 = old_s1 / (old_s1 + old_s0)
        s0 = old_s0 / (old_s1 + old_s0)

    if y1s1 == 0 and y1s0 == 0:
        sr = 1
    elif y1s0 == 0:
        sr = (y1s0/s0)/(y1s1/s1)
    elif y1s1 == 0:
        sr = (y1s1/s1)/(y1s0/s0)
    else:
        sr = min((y1s0/s0)/(y1s1/s1), (y1s1/s1)/(y1s0/s0)) 

    return float(sr)

def wl_accuracy(wl: Hypothesis, true_data: np.ndarray, fake_samples: np.ndarray, info: ContinuousDataInfo) -> float:
    true_features = true_data[:, info.feature_columns]
    fake_features = fake_samples[:, info.feature_columns]
    training_x, training_y = generate_adversarial_labels(true_features, fake_features)

    correct = (wl(training_x) > 0) == training_y
    return float(correct.float().mean())

def wl_loss(wl: TorchHypothesis, true_data: np.ndarray, fake_samples: np.ndarray, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], info: ContinuousDataInfo) -> float:
    true_features = true_data[:, info.feature_columns]
    fake_features = fake_samples[:, info.feature_columns]
    training_x, training_y = generate_adversarial_labels(true_features, fake_features)

    loss = loss(wl(training_x), training_y)
    return float(loss)

def density_plot(true_data: np.ndarray, fake_samples: np.ndarray, info: ContinuousDataInfo, save_name: str) -> None:
    true_features = true_data[:, info.feature_columns]
    fake_features = fake_samples[:, info.feature_columns]

    if true_features.shape[1] != 2:
        raise AttributeError("Density plot only works for 2D features")
    
    plt.scatter(true_features[:,0], true_features[:,1], c='blue', alpha=0.2)
    plt.scatter(fake_features[:,0], fake_features[:,1], c='red', alpha=0.2)
    plt.savefig(save_name)
    
def generate_prob_array(samples: np.ndarray, info: DiscreteDataInfo):
    values, counts = np.unique(samples, return_counts=True, axis=0)
    count_dict = defaultdict(int)
    for v, c in zip(values, counts):
        count_dict[tuple(v)] = c
        
    domain = info.full_domain

    return [count_dict[tuple(d)] / len(samples) for d in domain]

def generate_count_array(samples: np.ndarray, info: DiscreteDataInfo):
    values, counts = np.unique(samples, return_counts=True, axis=0)
    count_dict = defaultdict(int)
    for v, c in zip(values, counts):
        count_dict[tuple(v)] = c
        
    domain = info.full_domain

    return [count_dict[tuple(d)] for d in domain]

def sample_kl(data: np.ndarray, fake_samples: np.ndarray, info: DiscreteDataInfo, delta: float = 0) -> float:
    true_probs = np.array(generate_prob_array(data, info))
    fake_probs = np.array(generate_prob_array(fake_samples, info))
    
    return entropy(pk=true_probs + delta, qk=fake_probs + delta)

def kl(data: np.ndarray, density: DiscreteFairDensity, info: DiscreteDataInfo) -> float:
    true_probs = generate_prob_array(data, info)
    
    fake_probs = []
    for d in info.full_domain:
        cur_data = torch.tensor(d).reshape(1, -1)
        fake_probs.append(math.exp(density.log_prob(cur_data)))
    
    return entropy(pk=true_probs, qk=fake_probs)
 
def sklearn_hypothesis_acc(hypothesis, real_data: np.ndarray, fake_data: np.ndarray, info: Union[ContinuousDataInfo, DiscreteDataInfo]) -> float:
    x_data, y_data = generate_adversarial_labels(real_data, fake_data)
    return hypothesis.original_object.score(x_data, y_data)

def sklearn_accuracy(clf, data: np.ndarray, info: Union[ContinuousDataInfo, DiscreteDataInfo], non_sens: bool = False) -> float:
    if non_sens:
        features = data[:, info.feature_columns]
    else:
        features = data[:, info.unlabel_columns]
    labels = data[:, info.label_columns]
    
    predict = clf.predict(features)
    pscore = np.mean(labels.ravel() == predict)
    
    return pscore

def sklearn_equal_opportunity(clf, data: np.ndarray, info: Union[ContinuousDataInfo, DiscreteDataInfo], non_sens: bool = False) -> float:
    if non_sens:
        features = data[:, info.feature_columns]
    else:
        features = data[:, info.unlabel_columns]

    pred = clf.predict_proba(features)[:, 1].squeeze()
    if info.positive_label[0] == 0:
        pred = 1- pred
        
    sens_cols = data[:, info.sensitive_columns].squeeze()
    label_cols = data[:, info.label_columns].squeeze()

    predy1s1 = sum(pred * label_cols * sens_cols)
    predy1s0 = sum(pred * label_cols * (1 - sens_cols))

    y1s1 = sum(label_cols * sens_cols)
    y1s0 = sum(label_cols * (1 - sens_cols))

    if predy1s0 * y1s1 == 0:
        eo = (predy1s1 * y1s0) / (predy1s0 * y1s1)
    elif predy1s1 * y1s0 == 0:
        eo = (predy1s0 * y1s1) / (predy1s1 * y1s0)
    else:
        eo = min((predy1s1 * y1s0) / (predy1s0 * y1s1), (predy1s0 * y1s1) / (predy1s1 * y1s0))

    return float(eo)

def privilege_ratio(data: np.ndarray, info: Union[ContinuousDataInfo, DiscreteDataInfo]):
    pred_sens_vals = data[:, info.sensitive_columns]
    return np.sum(pred_sens_vals) / len(pred_sens_vals)

def generate_stat_summary(val_list: List[float], prefix = ''):
    return {
        prefix + 'min': np.min(val_list),
        prefix + 'mean': np.mean(val_list),
        prefix + 'max': np.max(val_list),
        prefix + 'diff': np.max(val_list) - np.min(val_list)
    }

def sklearn_kmean_privilege_ratio(kmeans, data: np.ndarray, info: Union[ContinuousDataInfo, DiscreteDataInfo]):
    pred_clusters = kmeans.predict(normalize(data[:, info.feature_columns + info.label_columns], axis=0))
    
    pr_vec = []
    for i in range(kmeans.n_clusters):
        in_clust = np.where(pred_clusters == i)
        pr_vec.append(privilege_ratio(data[in_clust], info))
    return generate_stat_summary(pr_vec, prefix = 'kmeans_pr_')

def sklearn_kmean_statistical_rate(kmeans, data: np.ndarray, info: Union[ContinuousDataInfo, DiscreteDataInfo]):
    pred_clusters = kmeans.predict(normalize(data[:, info.feature_columns + info.label_columns], axis=0))
    
    sr_vec = []
    for i in range(kmeans.n_clusters):
        in_clust = np.where(pred_clusters == i)
        sr_vec.append(statistical_rate(data[in_clust], info))

    return generate_stat_summary(sr_vec, prefix = 'kmeans_sr_')

def sklearn_kmean_mahalanobis(kmeans, data: np.ndarray, info: Union[ContinuousDataInfo, DiscreteDataInfo]) -> float:
    centers = kmeans.cluster_centers_
    loss = 0
    cur_data = normalize(data[:, info.feature_columns + info.label_columns], axis=0)
    for i in range(len(cur_data)):
        row = cur_data[i, :]
        loss += np.min(np.linalg.norm(centers - row, axis=1))

    return float(loss) / data.shape[0]

def w2_eval(true_data, fake_samples, n_bins):

    my_bins = ((true_data.shape[1] - 2) * [n_bins]) + [2, 2]
    
    fake_hist = np.histogramdd(fake_samples, bins=my_bins, density=True)
    true_hist = np.histogramdd(true_data, bins =fake_hist[1], density=True)

    centers = []
    for edges in true_hist[1]:
        cur_c = (edges[:-1] + edges[1:]) / 2
        centers.append(cur_c)

    pos_list = list(product(*reversed(centers)))

    dist_matrix = np.zeros((len(pos_list), len(pos_list)))
    for i in range(len(pos_list)):
        for j in range(i+1, len(pos_list)):
            cur_d = np.linalg.norm(np.array(pos_list[i]) - np.array(pos_list[j]))
            dist_matrix[i, j] = cur_d
            dist_matrix[j, i] = cur_d

    ot_dist = ot.emd2(true_hist[0].reshape(-1), fake_hist[0].reshape(-1), dist_matrix)
    return ot_dist

def kl_cts_eval(true_data, fake_samples, n_bins):

    my_bins = ((true_data.shape[1] - 2) * [n_bins]) + [2, 2]
    
    fake_hist = np.histogramdd(fake_samples, bins=my_bins, density=True)
    true_hist = np.histogramdd(true_data, bins =fake_hist[1], density=True)

    true_prob = true_hist[0].reshape(-1) + 1e-9
    fake_prob = fake_hist[0].reshape(-1) + 1e-9

    return entropy(pk=true_prob, qk=fake_prob)

def generate_recent_dt_dot(bdensity: DiscreteFairDensity, feature_names: List[str]):
    if len(bdensity.models) < 1:
        return None

    try:
        calibrated_m = bdensity.models[-1].prob_object.calibrated_classifiers_[0].base_estimator
    except:
        calibrated_m = None

    if bdensity.models[-1] is BaseDecisionTree:
        dt = bdensity.models[-1].prob_object
    elif isinstance(calibrated_m, DecisionTreeClassifier):
        dt = calibrated_m
    else:
        return None

    label_names = ['fake_example', 'true_example']
    hypothesis_features = feature_names

    return tree.export_graphviz(dt, out_file=None,
                                feature_names=hypothesis_features,
                                class_names=label_names, filled=True,
                                rounded=True, special_characters=True)


### Aggregated statistics functions
def discrete_result_statistics(booster, train_data, test_data, boost_i, kmeans_clusters, seed):
    boost_samples = booster.get_cur_samples(5 * len(train_data))
    info = train_data.info
    
    post_clf = DecisionTreeClassifier(max_depth=32)
    post_clf.fit(boost_samples[:, info.unlabel_columns], boost_samples[:, info.label_columns].ravel())
    
    post_kmeans = KMeans(n_clusters=kmeans_clusters, random_state=seed)
    post_kmeans.fit(normalize(boost_samples[:, info.feature_columns + info.label_columns], axis=0))
    
    clf_samples = copy.deepcopy(test_data.data)
    clf_samples[:, info.label_columns] = post_clf.predict_proba(test_data.data[:, info.unlabel_columns])[:, 1].reshape(-1, 1)
     
    clustering_dict = {}
    clustering_dict.update(sklearn_kmean_privilege_ratio(post_kmeans, test_data.data, info))
    clustering_dict.update(sklearn_kmean_statistical_rate(post_kmeans, test_data.data, info))
    clustering_dict['kmeans_dist'] = sklearn_kmean_mahalanobis(post_kmeans, test_data.data, info),

    return {
        'boost_iter': boost_i,
        'boosting': {
            'theta': booster.density.thetas[-1] if len(booster.density.thetas) > 0 else None,
            'norm': float(booster.density.tot_zs[-1]) if len(booster.density.tot_zs) > 0 else None,
            'wl_train_acc': float(booster.density.models[-1].wl_acc) if len(booster.density.models) > 0 else None,
            'wl_test_acc': sklearn_hypothesis_acc(booster.density.models[-1], test_data.data, boost_samples, info) if len(booster.density.models) > 0 else None,
            'wl_dot': generate_recent_dt_dot(booster.density, train_data.feature_names)
        },
        'data': {
            'rr': representation_rate(boost_samples, info),
            'sr': statistical_rate(boost_samples, info),
            'train_kl': kl(train_data.data, booster.density, info),
            'test_kl': kl(test_data.data, booster.density, info),
        },
        'prediction': {
            'clf_sr': statistical_rate(clf_samples, info),
            'clf_acc': sklearn_accuracy(post_clf, test_data.data, info),
            'clf_eo': sklearn_equal_opportunity(post_clf, test_data.data, info),
        },
        'clustering': clustering_dict,
    }

def continuous_result_statistics(booster, train_data, test_data, boost_i, kmeans_clusters, seed, n_bins=20):
    boost_samples = booster.get_cur_samples(5 * len(train_data))
    info = train_data.info

    post_clf = DecisionTreeClassifier(max_depth=32)
    post_clf.fit(boost_samples[:, info.unlabel_columns], boost_samples[:, info.label_columns].ravel())

    base_clf = DecisionTreeClassifier(max_depth=32)
    base_clf.fit(train_data.data[:, info.unlabel_columns], train_data.data[:, info.label_columns].ravel())

    post_kmeans = KMeans(n_clusters=kmeans_clusters, random_state=seed)
    post_kmeans.fit(normalize(boost_samples[:, info.feature_columns + info.label_columns], axis=0))

    clf_samples = copy.deepcopy(test_data.data)
    clf_samples[:, info.label_columns] = post_clf.predict_proba(test_data.data[:, info.unlabel_columns])[:, 1].reshape(-1, 1)

    clustering_dict = {}
    clustering_dict.update(sklearn_kmean_privilege_ratio(post_kmeans, test_data.data, info))
    clustering_dict.update(sklearn_kmean_statistical_rate(post_kmeans, test_data.data, info))
    clustering_dict['kmeans_dist'] = sklearn_kmean_mahalanobis(post_kmeans, test_data.data, info),

    return {
        'boost_iter': boost_i,
        'boosting': {
            'theta': booster.density.thetas[-1] if len(booster.density.thetas) > 0 else None,
            'norm': float(booster.density.joint_zs[-1]) if len(booster.density.joint_zs) > 0 else None,
            'wl_train_acc': float(booster.density.models[-1].wl_acc) if len(booster.density.models) > 0 else None,
            'wl_train_loss': float(booster.density.models[-1].wl_loss) if len(booster.density.models) > 0 else None,
        },
        'data': {
            'rr': representation_rate(boost_samples, info),
            'sr': statistical_rate(boost_samples, info),
            'train_wass2': w2_eval(train_data.data, boost_samples, n_bins),
            'test_wass2': w2_eval(test_data.data, boost_samples, n_bins),
            'train_kl': kl_cts_eval(train_data.data, boost_samples, n_bins),
            'test_kl': kl_cts_eval(test_data.data, boost_samples, n_bins),
        },
        'prediction': {
            'clf_sr': statistical_rate(clf_samples, info),
            'clf_acc': sklearn_accuracy(post_clf, test_data.data, info),
            'clf_eo': sklearn_equal_opportunity(post_clf, test_data.data, info),
        },
        'clustering': clustering_dict,
    }
