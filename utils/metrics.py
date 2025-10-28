import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error


def calc_recall(rank, ground_truth, k):
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(hit, k):
    hit = np.asarray(hit)[:k]
    return np.mean(hit)


def precision_at_k_batch(hits, k):
    res = hits[:, :k].mean(axis=1)
    return res


def average_precision(hit, cut):
    hit = np.asarray(hit)
    precisions = [precision_at_k(hit, k + 1) for k in range(cut) if len(hit) >= k]
    if not precisions:
        return 0.
    return np.sum(precisions) / float(min(cut, np.sum(hit)))


def dcg_at_k(rel, k):
    rel = np.asfarray(rel)[:k]
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return dcg


def ndcg_at_k(rel, k):
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def ndcg_at_k_batch(hits, k):
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def recall_at_k(hit, k, all_pos_num):
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def recall_at_k_batch(hits, k):
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res


def F1(pre, rec):
    pre = np.asarray(pre)
    rec = np.asarray(rec)
    denom = pre + rec
    f1 = np.where(denom > 0, (2.0 * pre * rec) / denom, 0.0)
    if f1.shape == ():
        return float(f1)
    return f1


def calc_auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def logloss(ground_truth, prediction):
    logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
    return logloss


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks):

    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for idx, u in enumerate(user_ids):
        train_pos_item_list = train_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[idx][train_pos_item_list] = -np.inf
        test_pos_item_binary[idx][test_pos_item_list] = 1

    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k)
        metrics_dict[k]['recall']    = recall_at_k_batch(binary_hit, k)
        metrics_dict[k]['f1']        = F1(metrics_dict[k]['precision'], metrics_dict[k]['recall'])
        metrics_dict[k]['ndcg']      = ndcg_at_k_batch(binary_hit, k)

    return metrics_dict

