from os import path

DATA_DIR = './data'
TRAINX = path.join(DATA_DIR, 'train_X.npy')
TRAINY = path.join(DATA_DIR, 'train_y.npy')
TESTX = path.join(DATA_DIR, 'test_X.npy')
train_num = 12399
test_num = 9801
# train_num = 100
# test_num = 100
nb_classes = 24

shape = (32, 32)
