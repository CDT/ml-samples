from mnist import MNIST
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random

# 1. 载入MNIST
print("1. 载入MNIST...")
mndata = MNIST('dataset')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
# 随机显示一个手写图
print("载入完成。随机显示一条训练集数据：")
print(mndata.display(
  train_images[random.randrange(0, len(train_images))]
))

# 2. 创建神经网络模型
# TODO 测试特征规范化的影响
# print("进行特征规范化...")
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# print("特征规范化完成。")
print("2. 创建神经网络（类型为Multi-layer Classifier，中间层维度：(5, 5, 5），最大迭代数1000）")
mlp = MLPClassifier(hidden_layer_sizes=5, max_iter=10, verbose=2)
print('神经网络训练中...')
mlp.fit(train_images, train_labels)
print('神经网络已训练完成。')
print('开始预测测试集...')
predictions = mlp.predict(test_images)
print('已预测测试集。')
print('评估神经网络模型：')
print('Confusion Matrix评估结果：')
print(confusion_matrix(test_labels.tolist(), predictions))
print('Classification Report评估结果：')
print(classification_report(test_labels.tolist(), predictions))
