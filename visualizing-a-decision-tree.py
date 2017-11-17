import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
print "feature_names"
print iris.feature_names
print "target_names"
print iris.target_names
print "First data"
print iris.data[0]
print "First target"
print iris.target[0]

test_idx = [0, 50, 100]

# training data
training_target = np.delete(iris.target, test_idx)
training_data = np.delete(iris.data, test_idx, axis = 0)
	
# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(training_data, training_target)

print "Prediction",	clf.predict(test_data)
print "Target",	test_target

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("iris")