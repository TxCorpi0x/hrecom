import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score
from common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)

TOP_K_ERROR = "k for ranking metric should be smaller than top_k of recommendation."


def auc_score(model, ratings, min_rate_value):
    """
    computes area under the ROC curve (AUC).
    The full name should probably be mean
    auc score as it is computing the auc
    for every user's prediction and actual
    interaction and taking the average for
    all users

    Parameters
    ----------
    model : BPR instance
        Trained BPR model

    ratings : scipy sparse csr_matrix, shape [n_users, n_items]
        sparse matrix of user-item interactions

    Returns
    -------
    auc : float 0.0 ~ 1.0
    """

    auc = 0.0
    ndcg = 0.0
    n_users, n_items = ratings.shape
    for user, row in enumerate(ratings):
        y_pred = model._predict_user(user)

        y_true = np.zeros(n_items)
        y_true[row.indices] = 1

        y_pred_b = np.zeros(n_items)
        y_pred_b[y_pred > min_rate_value] = 1

        try:
            auc += roc_auc_score(y_true, y_pred)
        except ValueError:
            pass
        # print(y_preds)
        ndcg += ndcg_score([y_true], [y_pred_b])

    auc /= n_users
    ndcg /= n_users

    return auc, ndcg


def get_user_hit(user_top_k, item_ids):
    hit = item_ids.isin(user_top_k).sum()
    return hit


def user_precision_at_k(user_top_k, item_ids, k):
    user_top_k = user_top_k[0, :k]
    hit = get_user_hit(user_top_k, item_ids)
    return hit / k


def user_recall_at_k(user_top_k, item_ids, k):
    user_top_k = user_top_k[0, :k]
    hit = get_user_hit(user_top_k, item_ids)
    n_items = item_ids.shape[0]
    return hit / n_items


def user_ndcg_at_k(user_top_k, item_ids, k):
    user_top_k = user_top_k[0, :k]
    minimum = min(len(item_ids), k)
    up = np.sum(np.isin(user_top_k, item_ids) * (1 / np.log2(np.arange(k) + 2)))
    down = np.sum(1 / np.log2(np.arange(minimum) + 2))
    return up / down


def precision_at_k(top_k_recommend, user_item_ids, k=None):
    top_k = top_k_recommend.shape[1]

    if k is None:
        k = top_k_recommend.shape[1]
    elif top_k < k:
        print(TOP_K_ERROR)
        return None

    top_k_users = top_k_recommend.index
    user_ids = user_item_ids[DEFAULT_USER_COL].unique()

    precision_list = []
    for u in user_ids:
        user_top_k = top_k_recommend[top_k_users == u].to_numpy()
        item_ids = user_item_ids.loc[
            user_item_ids[DEFAULT_USER_COL] == u, DEFAULT_ITEM_COL
        ]
        precision_list.append(user_precision_at_k(user_top_k, item_ids, k))

    return np.array(precision_list).mean()


def recall_at_k(top_k_recommend, user_item_ids, k=None):
    top_k = top_k_recommend.shape[1]

    if k is None:
        k = top_k_recommend.shape[1]
    elif top_k < k:
        print(TOP_K_ERROR)
        return None

    top_k_users = top_k_recommend.index
    user_ids = user_item_ids[DEFAULT_USER_COL].unique()

    recall_list = []
    for u in user_ids:
        user_top_k = top_k_recommend[top_k_users == u].to_numpy()
        item_ids = user_item_ids.loc[
            user_item_ids[DEFAULT_USER_COL] == u, DEFAULT_ITEM_COL
        ]
        recall_list.append(user_recall_at_k(user_top_k, item_ids, k))

    return np.array(recall_list).mean()


def ndcg_at_k(top_k_recommend, user_item_ids, k=None):
    top_k = top_k_recommend.shape[1]

    if k is None:
        k = top_k_recommend.shape[1]
    elif top_k < k:
        print(TOP_K_ERROR)
        return None

    top_k_users = top_k_recommend.index
    user_ids = user_item_ids[DEFAULT_USER_COL].unique()

    ndcg_list = []
    for u in user_ids:
        user_top_k = top_k_recommend[top_k_users == u].to_numpy()
        item_ids = user_item_ids.loc[
            user_item_ids[DEFAULT_USER_COL] == u, DEFAULT_ITEM_COL
        ]
        ndcg_list.append(user_ndcg_at_k(user_top_k, item_ids, k))

    return np.array(ndcg_list).mean()


def ranking_metrics(top_k_recommend, user_item_ids, k=None):
    return {
        "Precision@k": precision_at_k(top_k_recommend, user_item_ids, k),
        "Recall@k": recall_at_k(top_k_recommend, user_item_ids, k),
        "NDCG@k": ndcg_at_k(top_k_recommend, user_item_ids, k),
    }


eval_ranking_metrics = {
    "precision": lambda top_k_recommend, user_item_ids, k: precision_at_k(
        top_k_recommend, user_item_ids, k
    ),
    "recall": lambda top_k_recommend, user_item_ids, k: recall_at_k(
        top_k_recommend, user_item_ids, k
    ),
    "ndcg": lambda top_k_recommend, user_item_ids, k: ndcg_at_k(
        top_k_recommend, user_item_ids, k
    ),
}
