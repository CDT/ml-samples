import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. 数据获取与初始化
print('1. 读取数据：https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv(url, names=names)
print('读取完成。')
print('数据示例：')
irisdata.head()

# 设置X与y
X = irisdata.iloc[:, 0:4]
y = irisdata.select_dtypes(include=[object])
print('需要将y从字符串标签转换为数字标签')
print('转换前的y：')
y.Class.unique()
le = preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)
print('转换后的y：')
print(y)
# 配置训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print('已配置训练集和测试集，测试集占比为20%。')
# 特征规范化
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print('已使用StandardScaler将特征规范化。\n')

# 2. 创建神经网络并评估
print('2. 创建神经网络')
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())
print('神经网络已训练完成（模型：Multi-layer Perceptron 中间层：10，10，10 最大迭代数：1000）')
predictions = mlp.predict(X_test)
print('已预测测试集。')
print('评估神经网络模型：')
print('Confusion Matrix评估结果：')
print(confusion_matrix(y_test, predictions))
print('Classification Report评估结果：')
print(classification_report(y_test, predictions))

