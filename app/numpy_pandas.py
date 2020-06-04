import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
from sklearn.cluster import KMeans
import pymysql

db = pymysql.connect("localhost", "root", "root123", "ai2")
# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor(pymysql.cursors.DictCursor)
sql = 'updata person set age = 12 where id = 1;'
cursor.execute(sql)

# 使用 fetchone() 方法获取单条数据.
data = cursor.fetchall()
print(data)
# 关闭数据库连接
db.close()
# data = [[i['car'],i['car']] for i in data]
# data = np.array(data)
# plt.scatter(data[:, 0], data[:, 1], marker='o')
# plt.show()
#
# y_pred = KMeans(n_clusters=2).fit_predict(data)
# plt.scatter(data[:, 0], data[:, 1], c=y_pred)
# plt.savefig('C:\\Users\\yql\\PycharmProjects\\ai\\static\\images\\a.png')
# plt.show()