import tensorflow as tf
import numpy as np
import cv2, math

class Dataset():
    def __init__(self, dataset='mnist', dsize=None, x_normalization=True, y_one_hot=False, verbose=True):
        assert dataset in {'mnist','fashion_mnist','cifar10','cifar100'}, f'{dataset}不存在'
        self.dataset_name = dataset
        self.x_normalization, self.y_one_hot = x_normalization, y_one_hot
        if dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train, x_test = np.expand_dims(x_train,-1), np.expand_dims(x_test,-1)
            self.class_labels=np.array(["0","1","2","3","4","5","6","7","8","9"])
        elif dataset == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            x_train, x_test = np.expand_dims(x_train,-1), np.expand_dims(x_test,-1)
            self.class_labels=np.array(["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
        elif dataset == 'cifar10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            y_train, y_test = y_train.squeeze(), y_test.squeeze()
            self.class_labels=np.array(["airplain","automobile","bird","cat","deer","dog","frog","horse","ship","truck"])
        elif dataset == 'cifar100':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
            y_train, y_test = y_train.squeeze(), y_test.squeeze()
            self.class_labels=np.array(['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard',
                    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'])
        self.num_class = max(y_train)+1
        if dsize is not None and len(dsize) == 2:
            x_train = warpPolar(x_train, dsize)
            x_test = warpPolar(x_test, dsize)
        if x_normalization:
            x_train = x_train.astype(np.float32)/255.0
            x_test  = x_test.astype(np.float32)/255.0
        if y_one_hot:
            y_train = tf.keras.utils.to_categorical(y_train, self.num_class)
            y_test = tf.keras.utils.to_categorical(y_test, self.num_class)
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        _, self.H, self.W, self.C = self.x_train.shape
        if verbose:
            print(f'dataset={dataset}, num_class={self.num_class}')
            print(f'[Train] images: shape={x_train.shape}, dtype={x_train.dtype}; labels: shape={y_train.shape}, dtype={y_train.dtype}')
            print(f'[Test]  images: shape={x_test.shape}, dtype={x_test.dtype}; labels: shape={y_test.shape}, dtype={y_test.dtype}')
    def load_data(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
    def get_train(self):
        return self.x_train, self.y_train
    def get_test(self):
        return self.x_test, self.y_test
    def get_input_shape(self):
        return (self.H, self.W, self.C)
    def get_HW(self):
        return (self.H, self.W)

def warpPolar(images, dsize=(17, 360)):
    # dsize = (num_radius, num_theta)
    # imgs.shape = [im_NUM, num_theta, num_radius, im_C]
    im_NUM, im_H, im_W, im_C = images.shape
    im_dx, im_dy = im_W-1, im_H-1
    im_rdx, im_rdy = im_dx/2, im_dy/2
    im_center = im_rdx, im_rdy
    radius = math.sqrt(pow(im_rdx, 2)+pow(im_rdy, 2))
    padding_w, padding_h = math.ceil(radius-im_rdx), math.ceil(radius-im_rdy)
    imp_center = im_rdx+padding_w, im_rdy+padding_h
    imgs = np.empty([im_NUM,dsize[1],dsize[0],im_C], dtype=np.uint8)
    for i, img in enumerate(images):
        img = img.squeeze()
        img = cv2.copyMakeBorder(img,padding_h,padding_h,padding_w,padding_w,cv2.BORDER_REPLICATE)
        img = cv2.warpPolar(img, dsize, imp_center, radius, cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        imgs[i] = img
    return imgs