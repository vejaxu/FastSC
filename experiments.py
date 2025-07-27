from typing import List, Set
import pickle
import time
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.sparse
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import fetch_openml
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances

import stag.random
import stag.graph
import stag.cluster
import stag.graphio

import clusteralgs
import main

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, confusion_matrix

import h5py


ALGS_TO_COMPARE = {
    # "k eigenvectors": clusteralgs.spectral_cluster,
    # "log(k) eigenvectors": clusteralgs.spectral_cluster_logk,
    "log(k) PM": clusteralgs.fast_spectral_cluster,
    # "k PM": clusteralgs.spectral_cluster_pm_k,
    # "KASP": clusteralgs.KASP,
}


def align_labels(true_labels, pred_labels):
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-cm)  # 使用负号使得求最大匹配

    # 创建映射关系
    label_mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # 重新映射预测标签
    aligned_pred = np.array([label_mapping[label] for label in pred_labels])

    return aligned_pred


def print_performances(performances):
    for alg_name, perf in performances.items():
        print(f"{alg_name: >20}: "
              f"\tnmi: {perf.nmi: .3f} +/- {perf.nmi_std: .3f}", 
              f"\tari: {perf.ari: .3f} +/- {perf.ari_std: .3f}", 
              f"\tf1: {perf.f1: .3f} +/- {perf.f1_std: .3f}",
              f"\ttime: {perf.time: .3f}s +/- {perf.t_std: .3f}"
              )
    print()


def evaluate_one_algorithm(g: stag.graph.Graph,
                           k: int,
                           gt_labels: List[int],
                           method,
                           t_const=None,
                           data=None):
    """
    Evaluate the performance of a single spectral clustering algorithm.

    :param g: the graph to be clusters
    :param k: the number of clusters to find
    :param gt_labels: the ground truth labels
    :param method: the spectral clustering method to be called
    :return: a PerfData object with the results of running the algorithm
    """
    start = time.time()
    if method in [clusteralgs.spectral_cluster_pm_k, clusteralgs.fast_spectral_cluster]:
        labels = method(g, k, t_const=t_const)
    elif method in [clusteralgs.KASP, clusteralgs.nystrom_spectral_clustering]:
        labels = method(data, k, t_const)
    else:
        labels = method(g, k)
    end = time.time()

    # labels = labels[: len(gt_labels)]


    running_time = end - start

    # ari = stag.cluster.adjusted_rand_index(gt_labels, labels)
    nmi = normalized_mutual_info_score(gt_labels, labels)
    ari = adjusted_rand_score(gt_labels, labels)
    labels = align_labels(gt_labels, labels)
    f1 = f1_score(gt_labels, labels, average='macro')
    return main.PerfData(g, ari, nmi, f1, running_time)


def compare_algs(g: stag.graph.Graph, k: int, gt_labels: List[int],
                 algs_to_run=None, num_trials=1, t_const=None, data=None):
    """
    Compare the spectral clustering on the given graph.

    Optionally specify a dictionary with boolean values of which algorithms
    to be compared.
    """
    if algs_to_run is None:
        algs_to_run = {alg: True for alg in ALGS_TO_COMPARE.keys()}

    # If the data is not provided, we do not run the KASP algorithm.
    # This is because the KASP algorithm fundamentally requires the 'raw' data
    # and does not operate directly on a graph.
    if data is None and 'KASP' in algs_to_run:
        algs_to_run['KASP'] = False

    performances = {}

    # Initialise the necessary matrices on the stag graph object for fair
    # comparison
    mat = g.normalised_laplacian()
    mat = g.normalised_signless_laplacian()

    for alg_name, method in ALGS_TO_COMPARE.items():
        if alg_name in algs_to_run and algs_to_run[alg_name]:
            print(f"Running method: {alg_name}", end='..')
            aris = []
            nmis = []
            f1s = []
            times = []
            for t in range(num_trials):
                # print(f".", end='')
                print(f'epoch: {t}')
                this_perf = evaluate_one_algorithm(g, k, gt_labels, method, t_const=t_const, data=data)
                aris.append(this_perf.ari)
                nmis.append(this_perf.nmi)
                f1s.append(this_perf.f1)
                times.append(this_perf.time)
            print()
            performances[alg_name] = main.PerfData(g,
                                                   np.mean(aris),
                                                   np.mean(nmis),
                                                   np.mean(f1s),
                                                   np.mean(times),
                                                   ari_std=np.std(aris),
                                                   nmi_std=np.std(nmis),
                                                   f1_std=np.std(f1s),
                                                   t_std=np.std(times))

    return performances


def preprocess_openml_data(dataset_name: str):
    if dataset_name == "skin":
            with h5py.File(f"../data/{dataset_name}.mat", 'r') as file:
                X = file['fea'][()]
                y = file['gt'][()]
                X = X.T
                y = y.T
                X = np.array(X)
                y = np.array(y).flatten()
    else:
        file_path = f"../data/{dataset_name}.mat"
        data = scipy.io.loadmat(file_path)
        if dataset_name == 'pendigits':
            X = np.array(data['X'])
            y = np.array(data['gtlabels']).flatten()
        elif dataset_name == "landsat" or dataset_name == "cure-t2-4k" or dataset_name == "waveform3":
            X = np.array(data['data'])
            y = np.array(data['label']).flatten()
        elif dataset_name == "letters" or dataset_name == "skin":
            X = np.array(data['fea'])
            y = np.array(data['gt']).flatten()
        else: 
            X = np.array(data['data'])
            y = np.array(data['class']).flatten()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    target_to_label = {}
    gt_labels = []
    next_label = 0
    for label in y:
        if label not in target_to_label:
            target_to_label[label] = next_label
            next_label += 1
        gt_labels.append(target_to_label[label])


    # with h5py.File(f"../data/noise_data/{dataset_name}.mat", 'r') as file:
    #     # print("Keys in the file:", list(file.keys()))
    #     X = file['data_with_noise'][()]
    #     y = file['class'][()]
    #     X = X.T
    #     y = y.flatten()
    #     print(f"X.shape: {X.shape}")
    #     print(f"y.shape: {y.shape}")
    #     print(type(y))
    #     target_to_label = {}
    #     gt_labels = []
    #     next_label = 0
    #     for label in y:
    #         if label not in target_to_label:
    #             target_to_label[label] = next_label
    #             next_label += 1
    #         gt_labels.append(target_to_label[label])
    #     print(len(gt_labels))


    # mnist = fetch_openml(dataset_name)
    # replace_dict = {chr(i): i-96 for i in range(97, 107)}
    # X = np.array(mnist.data.replace(replace_dict))
    # target_to_label = {}
    # gt_labels = []
    # next_label = 0
    # for l in list(mnist.target):
    #     if l not in target_to_label:
    #         target_to_label[l] = next_label
    #         next_label += 1
    #     gt_labels.append(target_to_label[l])
    
    n = X.shape[0] - 1
    knn_graph = kneighbors_graph(X, n_neighbors=10, mode='connectivity',
                                 include_self=False)
    new_adj = scipy.sparse.lil_matrix(knn_graph.shape)
    for i, j in zip(*knn_graph.nonzero()):
        new_adj[i, j] = 1
        new_adj[j, i] = 1
    with open(f"data/{dataset_name}.pickle", 'wb') as fout:
        pickle.dump((new_adj, gt_labels), fout)
    with open(f"data/{dataset_name}_data.pickle", 'wb') as fout:
        pickle.dump(X, fout) 

    # dist_matrix = euclidean_distances(X, X)  # 形状为 (n, n)
    # sigma = np.mean(dist_matrix)
    # sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    # adj = scipy.sparse.lil_matrix(sim_matrix)
    # with open(f"data/{dataset_name}.pickle", 'wb') as fout:
    #     pickle.dump((adj, gt_labels), fout)
    # with open(f"data/{dataset_name}_data.pickle", 'wb') as fout:
    #     pickle.dump(X, fout)


def preprocess_data_if_needed(dataset_name):
    """
    Check whether the data file for the given data exists
    already. If not, then call the preprocessing function.
    """
    # if not os.path.isfile(f"data/{dataset_name}.pickle"):
    preprocess_openml_data(dataset_name)


def openml_experiment(dataset_name: str, t_const=15):
    preprocess_data_if_needed(dataset_name)

    with open(f"data/{dataset_name}.pickle", 'rb') as fin:
        adj, gt_labels = pickle.load(fin)
    with open(f"data/{dataset_name}_data.pickle", 'rb') as fin:
        X = pickle.load(fin)
    g = stag.graph.Graph(adj)

    # Compare the algorithms
    k = max(gt_labels) + 1
    num_trials = 10
    performances = compare_algs(g, k, gt_labels, num_trials=num_trials,
                                t_const=t_const,
                                data=X)

    print(f"\n Summary for {dataset_name} graph, n = {g.number_of_vertices()}, k = {k}\n")
    print_performances(performances)