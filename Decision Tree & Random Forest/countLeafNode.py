import joblib
import numpy as np

max_leaf = [10,100,1000,10000,100000,1000000,10000000,100000000]


# This is referenced from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
for max_num in max_leaf:
    # load, no need to initialize the loaded_rf
    decisionTree = joblib.load("decisionTree_%d.joblib" % max_num)

    n_nodes = decisionTree.tree_.node_count
    children_left = decisionTree.tree_.children_left
    children_right = decisionTree.tree_.children_right
    feature = decisionTree.tree_.feature
    threshold = decisionTree.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    # Count number of leaf nodes
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    
    print(np.count_nonzero(is_leaves))

