from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from itertools import repeat
import glob
import os
from sklearn.metrics import precision_score, accuracy_score, mean_squared_error
from PIL import Image
import numpy as np

# Classes
left_paths = []
right_paths = []
straight_paths = []
up_paths = []

happy_paths = []
sad_paths = []
angry_paths = []
neutral_paths = []

eyes_open_paths = []
sunglass_paths = []

# Listing paths of files for each class
for folder in os.listdir('faces_4'):
    # head position
    # left_paths += np.array(Image.open('faces_4/' + folder + '/*left*.pgm'))
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
    eyes_open_paths += glob.glob('faces_4/' + folder + '/*eyes_open*.pgm')
    sunglass_paths += glob.glob('faces_4/' + folder + '/*sunglass*.pgm')

filenames = []
filenames.extend([left_paths, right_paths, straight_paths, up_paths, happy_paths, sad_paths, angry_paths, neutral_paths, eyes_open_paths, sunglass_paths])

left = []
right = []
straight = []
up = []
happy = []
sad = []
angry = []
neutral = []
eyes_open = []
sunglass = []

classes = []
classes.extend([left, right, straight, up, happy, sad, angry, neutral, eyes_open, sunglass])
i=0
for category in filenames:
    for filename in category:
        img = Image.open(filename)
        # PpmImageFile to Numpy Array
        classes[i].append( np.array(img).flatten())
        img.close()
    i+=1

# for item in left_paths:
#     left.append(Image.open(item))


head_x = left + right + straight + up
emotion_x = happy + sad + angry + neutral
eyes_x = eyes_open + sunglass

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
eyes_y.extend(repeat(0, len(eyes_open)))
eyes_y.extend(repeat(1, len(sunglass)))

features = []
features.extend([head_x, emotion_x, eyes_x])

# Splitting train and test datasets
head_x_train, head_x_test, head_y_train, head_y_test = train_test_split(head_x, head_y, test_size = 0.3)
emotion_x_train, emotion_x_test, emotion_y_train, emotion_y_test = train_test_split(emotion_x, emotion_y, test_size=0.3)
eyes_x_train, eyes_x_test, eyes_y_train, eyes_y_test = train_test_split(eyes_x, eyes_y, test_size=0.3)

classifier = SVC(gamma = 0.001)
classifier.fit(head_x_train, head_y_train)

y_predicted = classifier.predict(head_x_test)

print(y_predicted, head_y_test)
print(y_predicted == head_y_test)
# mse = mean_squared_error(head_y_test, y_predicted)
# prec = precision_score(head_y_test, y_predicted, average='macro')
# acc = accuracy_score(head_x_test, y_predicted)
# print('mean squared error:', mse)
# print('precision: ', prec)
# print('accuracy: ', acc)