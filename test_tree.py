from sklearn import tree
from sklearn.model_selection import train_test_split


features = [
	[3, 11],
	[4, 11],
	[150, 22],
	[170, 22],
	[555, 33],
	[600, 33]
]
labels = [0, 0, 1, 1, 2, 2]
clf = tree.DecisionTreeClassifier()

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.5)

clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)


print "features Train: ", features_train, ", Test: ", features_test
print "labels Train: ", labels_train, ", Test: ", labels_test
print "Prediction for features_test is: ", prediction, ", labels are: ", labels_test
from sklearn.metrics import accuracy_score

print accuracy_score(labels_test, prediction)


# Custom test
custom_features = [
	[800, 444],
	[1, 22],
	[220, 1]
]

custom_labels = [2, 0, 1]

prediction2 = clf.predict(custom_features)

print accuracy_score(custom_labels, prediction2)

