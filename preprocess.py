from PIL import Image, ImageCms
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
from config import DATA_DIR, TRAINX, TRAINY, TESTX, train_num, test_num, shape

srgb_profile = ImageCms.createProfile("sRGB")
lab_profile  = ImageCms.createProfile("LAB")
rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")


def covert2lab(im):
    return ImageCms.applyTransform(im, rgb2lab_transform)


def load_images(file_list):
    global DATA_DIR
    # imgs = [np.asarray(covert2lab(Image.open(path.join(DATA_DIR, f)).resize(shape))) for f in file_list]
    imgs = [np.asarray(Image.open(path.join(DATA_DIR, f)).resize(shape)) for f in file_list]
    return np.asarray(imgs)


def load_tsv(file, file_list):
    file_path = path.join(DATA_DIR, file)
    df = pd.read_csv(file_path, delimiter='\t')
    category_dict = df.set_index('file_name').to_dict()['category_id']

    categorys = [category_dict[path.basename(f)] for f in file_list]
    return np.asarray(categorys)


if __name__ == '__main__':
    file_list = ['train/train_%d.jpg' % i for i in range(train_num)]
    train_X = load_images(file_list)

    train_y = load_tsv('train_master.tsv', file_list)

    file_list = ['test/test_%d.jpg' % i for i in range(test_num)]
    test_X = load_images(file_list)

    print(TRAINX, train_X.shape)
    print(TRAINY, train_y.shape)
    print(TESTX, test_X.shape)

    np.save(TRAINX, train_X)
    np.save(TRAINY, train_y)
    np.save(TESTX, test_X)

    # print(train_X)
    # print(train_y)

    # train_X = np.load(TRAINX)
    # train_y = np.load(TRAINY)
    # test_X = np.load(TESTX)
