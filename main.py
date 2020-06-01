from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from itertools import repeat
import glob
import os
from sklearn.metrics import precision_score, accuracy_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def best_parameters(X, y):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    return grid.best_params_

def print_metrics(classifier, y_test, predicted, class_names):
    print("Classification report for classifier %s:\n%s\n"
        % (classifier, metrics.classification_report(y_test, predicted, target_names=class_names)))

    print('Confusion Matrix')
    print( metrics.confusion_matrix(y_test, predicted) )


# Classes
left_paths = []
right_paths = []
straight_paths = []
up_paths = []

happy_paths = []
sad_paths = []
angry_paths = []
neutral_paths = []

open_paths = []
sunglasses_paths = []

# Listing paths of files for each class
for folder in os.listdir('faces_4'):
    # head position
    left_paths += glob.glob('faces_4/' + folder + '/*left*.pgm')
    right_paths += glob.glob('faces_4/' + folder + '/*right*.pgm')
    straight_paths += glob.glob('faces_4/' + folder + '/*straight*.pgm')
    up_paths += glob.glob('faces_4/' + folder + '/*up*.pgm')

    # emotion
    happy_paths += glob.glob('faces_4/' + folder + '/*happy*.pgm')
    sad_paths += glob.glob('faces_4/' + folder + '/*sad*.pgm')
    angry_paths += glob.glob('faces_4/' + folder + '/*angry*.pgm')
    neutral_paths += glob.glob('faces_4/' + folder + '/*neutral*.pgm')

    # eyes
    open_paths += glob.glob('faces_4/' + folder + '/*open*.pgm')
    sunglasses_paths += glob.glob('faces_4/' + folder + '/*sunglasses*.pgm')

paths = []
paths.extend([left_paths, right_paths, straight_paths, up_paths, happy_paths, sad_paths, angry_paths, neutral_paths, open_paths, sunglasses_paths])

left = []
right = []
straight = []
up = []
happy = []
sad = []
angry = []
neutral = []
_open = []
sunglasses = []

classes = []
classes.extend([left, right, straight, up, happy, sad, angry, neutral, _open, sunglasses])
i=0
for category in paths:
    for filename in category:
        img = plt.imread(filename)
        classes[i].append( img.flatten())
    i+=1

head_x = left + right + straight + up
emotion_x = happy + sad + angry + neutral
eyes_x = _open + sunglasses

head_y = []
head_y.extend(repeat(0, len(left)))
head_y.extend(repeat(1, len(right)))
head_y.extend(repeat(2, len(straight)))
head_y.extend(repeat(3, len(up)))

emotion_y = []
emotion_y.extend(repeat(0, len(happy)))
emotion_y.extend(repeat(1, len(sad)))
emotion_y.extend(repeat(2, len(angry)))
emotion_y.extend(repeat(3, len(neutral)))

eyes_y = []
eyes_y.extend(repeat(0, len(_open)))
eyes_y.extend(repeat(1, len(sunglasses)))

# Splitting train and test datasets
head_x_train, head_x_test, head_y_train, head_y_test = train_test_split(head_x, head_y, test_size = 0.3)
emotion_x_train, emotion_x_test, emotion_y_train, emotion_y_test = train_test_split(emotion_x, emotion_y, test_size=0.3)
eyes_x_train, eyes_x_test, eyes_y_train, eyes_y_test = train_test_split(eyes_x, eyes_y, test_size=0.3)

features_dic = {'head' : [ head_x_train, head_x_test, head_y_train, head_y_test ],
                'emotion' : [ emotion_x_train, emotion_x_test, emotion_y_train, emotion_y_test ],
                'eyes' : [ eyes_x_train, eyes_x_test, eyes_y_train, eyes_y_test ]}

class_dic = {'head': ['left', 'right', 'straight', 'up'], 
             'emotion' : ['happy', 'sad', 'angry', 'neutral'], 
             'eyes' : ['open', 'sunglasses'] }

# 
class_names = ['happy', 'sad', 'angry', 'neutral']
X = emotion_x_train
X_test = emotion_x_test
y = emotion_y_train
y_test = emotion_y_test
print('train')
print(len(y))
print('test')
print(len(y_test))



for key in features_dic:
    class_names = class_dic[key]
    feature = features_dic[key]
    X = feature[0]
    X_test = feature[1]
    y = feature[2]
    y_test = feature[3]

    params = best_parameters(X,y)

    pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=params['C'], gamma=params['gamma'], kernel='linear', cache_size=500))])
    pipe.fit(X,y)
    predicted = pipe.predict(X_test)

    print('predicted')
    print(len(predicted))

    print_metrics(pipe, y_test, predicted, class_names)


# Parameters found
# head position: C=10.0, gamma=1e-06
# emotion: C=0.01, gamma=1e-09 (Bad results)
# eyes: C=10000.0, gamma=1e-09 