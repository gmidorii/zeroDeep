# coding: utf-8
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
from common.functions import sigmoid,softmax


def image_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def print_img_sample(x_train, t_train):
    img = x_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    image_show(img)


def init_network():
    with open("scratch-data/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


# data load
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

# print_img_sample(x_train=x_train, t_train=t_train)

network = init_network()

accuracy_cnt = 0
for i in range(len(x_test)):
    y = predict(network, x_test[i])
    p = np.argmax(y)
    if p == t_test[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))
