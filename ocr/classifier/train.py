import pandas as pd
from sklearn.externals import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('dataset.csv', sep=',')

images = dataset.iloc[:, 1:]
labels = dataset.iloc[:, :1]

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.9, random_state=0)

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())

print(clf.score(test_images, test_labels))

joblib.dump(clf, 'clf_svm_def.pkl', 3)
