import numpy as np
import pandas as pd
import copy
import scipy.stats
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


"""
define basic functions for AFT1
"""
def search_path(estimator, y_threshold):
    """
    return path index list containing
    [{leaf node id, inequality symbol, threshold, feature index}].
    estimator: decision tree
    maxj: the number of selected leaf nodes
    """
    """ select leaf nodes whose outcome is aim_label """
    # information of left child node
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    # leaf nodes ID
    leaf_nodes = np.where(children_left == -1)[0]
    leaf_nodes = [
        i for i in leaf_nodes if estimator.tree_.value[i] >= y_threshold]
    """ search the path to the selected leaf node """
    paths = {}
    for leaf_node in leaf_nodes:
        """ correspond leaf node to left and right parents """
        child_node = leaf_node
        parent_node = -100  # initialize
        parents_left = []  # 左側親ノード
        parents_right = []  # 右側親ノード
        while (parent_node != 0):
            if (np.where(children_left == child_node)[0].shape == (0, )):
                parent_left = -1  # 左側親ノードが存在しない場合は-1
                parent_right = np.where(
                    children_right == child_node)[0][0]
                parent_node = parent_right
            elif (np.where(children_right == child_node)[0].shape == (0, )):
                parent_right = -1  # 右側親ノードが存在しない場合は-1
                parent_left = np.where(children_left == child_node)[0][0]
                parent_node = parent_left
            parents_left.append(parent_left)
            parents_right.append(parent_right)
            """ for next step """
            child_node = parent_node
        # nodes dictionary containing left parents and right parents
        paths[leaf_node] = (parents_left, parents_right)
    path_info = {}
    for i in paths:
        node_ids = []  # node ids used in the current node
        # inequality symbols used in the current node
        inequality_symbols = []
        thresholds = []  # thretholds used in the current node
        features = []  # features used in the current node
        parents_left, parents_right = paths[i]
        for idx in range(len(parents_left)):
            if (parents_left[idx] != -1):
                """ the child node is the left child of the parent """
                node_id = parents_left[idx]  # node id
                node_ids.append(node_id)
                inequality_symbols.append(0)
                thresholds.append(threshold[node_id])
                features.append(feature[node_id])
            elif (parents_right[idx] != -1):
                """ the child node is the right child of the parent """
                node_id = parents_right[idx]
                node_ids.append(node_id)
                inequality_symbols.append(1)
                thresholds.append(threshold[node_id])
                features.append(feature[node_id])
            path_info[i] = {'node_id': node_ids[::-1],
                            'inequality_symbol': inequality_symbols[::-1],
                            'threshold': thresholds[::-1],
                            'feature': features[::-1]}
    return path_info


def esatisfactory_instance(x, epsilon, path_info):
    """
    return the epsilon satisfactory instance of x.
    """
    esatisfactory = copy.deepcopy(x)
    for feature_idx in np.unique(path_info['feature']):
        """ loop by each feature -- i is feature index (feature name). """
        positions = np.where(np.array(path_info['feature'])==feature_idx)[0]  # positions of path_information list
        theta_upp = float('inf')
        theta_low = -float('inf')
        for posi in positions:
            if path_info['inequality_symbol'][posi]==0:
                """ posiが大きいほど厳しい条件になるので，順番に更新していくだけで良い """
                theta_upp = path_info['threshold'][posi]  # upper bounded threshold
            elif path_info['inequality_symbol'][posi]==1:
                theta_low = path_info['threshold'][posi]  # lower bounded threshold
        if theta_low == -float('inf'):
            esatisfactory[feature_idx] = theta_upp - epsilon
        elif theta_upp == float('inf'):
            esatisfactory[feature_idx] = theta_low + epsilon
        else:
            esatisfactory[feature_idx] = (theta_low + theta_upp)/2
    return esatisfactory


def transam(ensemble_regressor, x, y_threshold, epsilon, cost_func):
    """
    x: feature vector
    y_threshold: threshold value to select the positive path
    """
    """ initialize """
    x_out = copy.deepcopy(x)  # initialize output
    delta_mini = 10**3  # initialize cost
    check_cnt = 0  # check whether epsilon satisfactory instance is updated
    for estimator in ensemble_regressor:
        if (ensemble_regressor.predict(x.reshape(1, -1)) < y_threshold
            and estimator.predict(x.reshape(1, -1)) < y_threshold):
            paths_info = search_path(estimator, y_threshold)
            for key in paths_info:
                """ generate epsilon-satisfactory instance """
                path_info = paths_info[key]
                es_instance = esatisfactory_instance(x, epsilon, path_info)
                if ensemble_regressor.predict(es_instance.reshape(1, -1)) > y_threshold:
                    if cost_func(x, es_instance) < delta_mini:
                        x_out = es_instance
                        delta_mini = cost_func(x, es_instance)
                        check_cnt += 1  # add update counter
            else:
                continue
    return x_out
