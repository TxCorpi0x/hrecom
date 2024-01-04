import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error


def mae_auc_score(model, ratings):
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

    # y_actual  = []
    # y_prediction  = []
    auc = 0.0
    rms = 0.0
    rmse = 0.0
    n_users, n_items = ratings.shape
    for user, row in enumerate(ratings):
        y_pred = model._predict_user(user)
        # y_prediction.append(y_pred)
        y_true = np.zeros(n_items)
        # y_actual= y_true[row.indices]
        y_true[row.indices] = 1

        auc += roc_auc_score(y_true, y_pred)
        # x= mean_squared_error(y_true, y_pred, squared=False)
        # if x >1:
        #     print(x)
        rms += mean_squared_error(y_true, y_pred, squared=False)

    auc /= n_users
    rms /= n_users

    # rms = mean_squared_error(y_actual, y_prediction, squared=False)
    return rms, auc