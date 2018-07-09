# https://www.youtube.com/watch?v=tNa99PG8hR8
# http://scikit-learn.org/stable/datasets/
# http://scikit-learn.org/stable/modules/tree.html#tree
# https://www.graphviz.org/download/

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

# viz code
import graphviz

# dot_data = StringIO()
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                impurity=False)

graph = graphviz.Source(dot_data)
graph.render("iris", view=True)

