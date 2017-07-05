import sys
from os import path
from keras.models import load_model
from config import TRAINX, TESTX
import numpy as np
import time

load_file = sys.argv[1] if len(sys.argv) > 1 else ''
if not path.exists(load_file):
    raise ValueError('No file: %s' % load_file)

print('load model', load_file)
model = load_model(load_file)


train_X = np.load(TRAINX)
# train_y = np.load(TRAINY)
test_X = np.load(TESTX)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

# subtract mean and normalize
# mean_image = np.mean(train_X, axis=0)
# train_X -= mean_image
# test_X -= mean_image
# train_X /= 128.
# test_X /= 128.

print(test_X.shape)

pred = model.predict(test_X, verbose=1)

print(pred.shape)

t = int(time.time())
f = open(path.join('output', 'predict_%d.csv' % t), 'w')
for i, c in enumerate(pred):
    f.write('test_%d.jpg, %d\n' % (i, np.argmax(c)))

f.close()
