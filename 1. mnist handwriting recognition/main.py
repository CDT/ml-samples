from mnist import MNIST
import random

# 1. 载入MNIST
mndata = MNIST('dataset')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
# 随机显示一个手写图
index = random.randrange(0, len(train_images))
print(mndata.display(train_images[index]))