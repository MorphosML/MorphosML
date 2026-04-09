import numpy as np 
from collections import Counter

# === Decision tree from scratch ===

class Node:
  def __init__(self,feature,threshold,left=None,right=None,*,value=None): # after the * u need to specify the value and that is a good thing cause value means it is a leaf

    self.feature = feature
    self.threshold = threshold
    self.right = right
    self.left = left
    self.value = value

  def is_leaf_node(self):
    return self.value is not None


class DecisionTree:
  def __init__(self,min_samples_split=2,max_depth=100,n_features=None):

    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    self.n_features = n_features
    self.root = None

  def fit(self,X,y):
     self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features) # safety measure
     self.root = self._grow_tree(X,y)


  def _grow_tree(self,X,y,depth=0):
    n_samples, n_feats = X.shape
    n_labels = len(np.unique(y))

    # check the stopping criteria
    if (
        depth >= self.max_depth
        or n_labels == 1
        or n_samples < self.min_samples_split
    ):
        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value)

    # choose random features
    feat_idxs = np.random.choice(n_feats,self.n_features,replace=False) # np.random.choice(a,size=None) ;)

    # find the best
    best_feature,best_thresh  = self._best_split(X,y,feat_idxs)

    # creating child nodes


  def _best_split(self,X,y,feat_idxs):
    best_again = -1
    split_idx, split_threshold = None, None

    for feature_idx in feat_idxs:
      X_column = X[:,feature_idx]
      thresholds = np.unique(X_column)

      for thr in thresholds:
        # calculate the information gain
        gain = self._information_gain()

        if gain > best_gain:
          best_gain = gain
          split_idx = feature_idx
          split_threshold = thr

    return split_idx,split_threshold # best split feature

  def _information_gain(self,X_column,y,threshold):
    """
    Summary : Implementing information gain based on the formula :
    IG = E(parent) - [weighted_avg] * E(children)

    Entropy (E) :
    E(S)=−∑pi​log2​(pi​)

    """

    # parent entropy
    parent_entropy = self._entropy(y)

    # create children
    left_idx,right_idx = self._split()

    # calculate the weighted avg

    # entropy of children

    # calculate the IG
    pass

  def _split(self,):
    pass


  def _entropy(self,y):
    hist = np.bincout(y) # counting each value for each time they appeared

    ps = hist/len(y)

    return -np.sum([ p * np.log(p) for p in ps if p > 0])


  def _most_common_label(self,y):
    counter = Counter(y)
    return counter.most_common(1)[0][0] # most common tuple and the info

  def predict(self,X,y):
    pass
