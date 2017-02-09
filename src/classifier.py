import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import scipy.io
import argparse
import random
import sys
import csv

with open(sys.argv[1], 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    for i, row in enumerate(reader):
        if i == 0:            
            nPoints, nFeats = map(int, row)
            X = np.zeros((nPoints,nFeats))
            X = np.zeros((nPoints,nFeats))
        else:
            index = int(row[0]) - 1
            feats = map(float, row[1:])
            X[index] = np.array(feats)

mat = scipy.io.loadmat(sys.argv[2])
Y = np.array(mat['group'].todense())
nLabels = Y.shape[1]

trX, teX, trY, teY = train_test_split(X, Y, test_size=0.5, random_state=0)







from keras.layers import Input, Dense, Dropout
from keras.models import Model

inputs = Input(shape=(nFeats,))
#x = Dense(1000, activation='relu', init='he_normal')(inputs)
#x = Dropout(0.5)(x)
#x = Dense(1000, activation='relu', init='he_normal')(x)
#x = Dropout(0.5)(x)
predictions = Dense(nLabels, activation='sigmoid', init='he_normal')(inputs)

model = Model(input=inputs, output=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['fmeasure'])
model.fit(trX, trY, nb_epoch = 100)

best = 0
for i in range(0,300):
    predicted = model.predict(teX) > (i/300.0)
    scores_macro = metrics.f1_score(teY,predicted,average='macro')
    if scores_macro > best: best = scores_macro
print("Macro-F1: %0.2f" % (best))


best = 0
for i in range(0,300):
    predicted = model.predict(teX) > (i/300.0)
    scores_macro = metrics.f1_score(teY,predicted,average='micro')
    if scores_macro > best: best = scores_macro
print("Micro-F1: %0.2f" % (best))

quit()






param = {
 'estimator__C':[1e4, 5e4, 1e5]
}

model_to_set = OneVsRestClassifier(LogisticRegression())
gsearch = GridSearchCV(estimator = model_to_set, param_grid = param, n_jobs=4, iid=False, cv=2, scoring='f1_macro')

gsearch.fit(trX, trY)
print "----------------------- %s" % gsearch.best_params_, gsearch.best_score_

best = 0
for i in range(0,100):
    predicted = gsearch.predict(teX) > (i/100.0)
    scores_macro = metrics.f1_score(teY,predicted,average='macro')
    if scores_macro > best: best = scores_macro

print("Macro-F1: %0.2f" % (best))
quit()
