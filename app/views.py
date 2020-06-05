
from django.http import HttpResponseRedirect, HttpResponse,JsonResponse
from django.shortcuts import render,reverse,redirect
# Django自带的用户验证,login
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.generic.base import View
from .forms import LoginForm,RegisterForm
# 进行密码加密
from django.contrib.auth.hashers import make_password
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy.types import NVARCHAR, Float, Integer
from django.core.paginator import Paginator
import os


class LoginView(View):
    def get(self,request):
        return render(request, 'login.html',)
    def post(self,request):
        login_form = LoginForm(request.POST)
        if login_form.is_valid():
            user_name = request.POST.get('username', '')
            pass_word = request.POST.get('password', '')
            # print(user_name)
            user = authenticate(username=user_name, password=pass_word)
            # print(user)
            if user is not None:
                login(request, user)
                return redirect(reverse('index'))
            else:
                return render(request, 'login.html', {'msg': '用户名或者密码错误'})
        else:
            return render(request, "login.html", {"login_form": login_form})
class RegisterView(View):
    def get(self, request):
        return render(request, "register.html")
    def post(self, request):
        # 实例化form
        register_form = RegisterForm(request.POST)
        if register_form.is_valid():
            user_name = request.POST.get("username", "")
            # 用户查重
            if User.objects.filter(username=user_name):
                return render(request, "register.html", { "msg": "用户已存在"})
            pass_word = request.POST.get("password", "")
            pass_word = make_password(pass_word)
            User.objects.create(username=user_name,password=pass_word)
            return HttpResponse('注册成功!')
        else:
            return render(request,'register.html',{'register_form':register_form})


class TestView(View):
    def get(self,request):
        return render(request,'test.html')
    def post(self,request):
        response = {'state': True}
        btn_num = request.POST.get('btn_num', '')
        time_step = request.POST.get('time_step', '')
        time_step = int(time_step)
        cell_num = request.POST.get('cell_num', '')
        cell_num = int(cell_num)
        iters = request.POST.get('iters', '')
        iters = int(iters)
        train_set = request.POST.get('train_set', '')
        train_set = float(train_set)
        import numpy as np
        from tensorflow import keras
        import matplotlib.pyplot as plt
        from keras.layers import Dense
        from keras.layers import LSTM
        import pandas as pd
        from keras.models import Sequential, load_model
        from sklearn.preprocessing import MinMaxScaler
        if btn_num == 'btn3':
            dataframe = pd.read_csv('./static/upload/airline.csv', usecols=[2], engine='python', skipfooter=3)
            dataset = dataframe.values
            dataset = dataset.astype('float32')
            plt.plot(dataset)
            plt.show()
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            train_size = int(len(dataset) * train_set)
            trainlist = dataset[:train_size]
            testlist = dataset[train_size:]

            def create_dataset(dataset, look_back):
                # 这里的look_back与timestep相同
                dataX, dataY = [], []
                for i in range(len(dataset) - look_back - 1):
                    a = dataset[i:(i + look_back)]
                    dataX.append(a)
                    dataY.append(dataset[i + look_back])
                return np.array(dataX), np.array(dataY)
            look_back = time_step
            trainX, trainY = create_dataset(trainlist, look_back)
            testX, testY = create_dataset(testlist, look_back)
            trainX = np.reshape(trainX, (trainX.shape[0], 2, 1))
            testX = np.reshape(testX, (testX.shape[0], 2, 1))
            # create and fit the LSTM network
            model = Sequential()
            model.add(LSTM(cell_num, input_shape=(None, 1)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(trainX, trainY, epochs=iters, batch_size=1, verbose=2)
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)
            # 反归一化
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform(trainY)
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform(testY)
            plt.plot(trainPredict[1:])
            plt.show()
            plt.plot(testY,'b',label='true')
            plt.plot(testPredict[1:],'r',label='predict')
            plt.xlabel('sample')
            plt.ylabel('people')
            plt.savefig('./static/images/yuce.png')
            plt.show()
            return JsonResponse(response)
        else:
            import numpy
            from tensorflow import keras
            import matplotlib.pyplot as plt
            from pandas import read_csv
            import math
            from keras.models import Sequential
            from keras.layers import Dense

            # convert an array of values into a dataset matrix
            def create_dataset(dataset, look_back=1):
                dataX, dataY = [], []
                for i in range(len(dataset) - look_back - 1):
                    a = dataset[i:(i + look_back), 0]
                    dataX.append(a)
                    dataY.append(dataset[i + look_back, 0])
                return numpy.array(dataX), numpy.array(dataY)

            # fix random seed for reproducibility
            numpy.random.seed(7)

            # load the dataset
            dataframe = read_csv('./static/upload/airline.csv', usecols=[2], engine='python', skipfooter=3)
            dataset = dataframe.values
            dataset = dataset.astype('float32')

            # split into train and test sets
            train_size = int(len(dataset) * 0.67)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

            # reshape dataset
            look_back = 3
            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(test, look_back)

            # create and fit Multilayer Perceptron model
            model = Sequential()
            model.add(Dense(12, input_dim=look_back, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

            # Estimate model performance
            trainScore = model.evaluate(trainX, trainY, verbose=0)
            print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
            testScore = model.evaluate(testX, testY, verbose=0)
            print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

            # generate predictions for training
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)

            # shift train predictions for plotting
            trainPredictPlot = numpy.empty_like(dataset)
            trainPredictPlot[:, :] = numpy.nan
            trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

            # shift test predictions for plotting
            testPredictPlot = numpy.empty_like(dataset)
            testPredictPlot[:, :] = numpy.nan
            testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
            plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

            # plot baseline and predictions
            plt.plot(dataset)
            plt.plot(trainPredictPlot)
            plt.plot(testPredictPlot)
            plt.xlabel('样本数')
            plt.ylabel('人数')
            plt.savefig('./static/images/yuce.png')
            plt.show()
            return JsonResponse(response)
class ProjectView(View):
    def get(self,request):

        return render(request,'project.html')
class DataView(View):
    def get(self,request):
        return render(request, 'data.html')
class TextProcessTwoView(View):
    def get(self,request):
        return render(request, 'text_process_two.html')
class CreateDataView(View):
    def get(self,request):
        return render(request, 'createData.html')
    def post(self,request):
        myFile = request.FILES.get("myFile", None)  # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("没有文件传入")
        import os
        destination = open(os.path.join("./static/upload", myFile.name), 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()
        if myFile.name.endswith('.csv'):
            engine = create_engine('mysql+pymysql://root:root123@localhost:3306/ai')
            # 建立连接
            con = engine.connect()
            df = pd.read_csv('./static/upload/'+myFile.name)
            def map_types(df):
                dtypedict = {}
                for i, j in zip(df.columns, df.dtypes):
                    if "object" in str(j):
                        dtypedict.update({i: NVARCHAR(length=255)})
                    if "float" in str(j):
                        dtypedict.update({i: Float(precision=2, asdecimal=True)})
                    if "int" in str(j):
                        dtypedict.update({i: Integer()})
                return dtypedict
            dtypedict = map_types(df)
            # 通过dtype设置类型 为dict格式{“col_name”:type}
            tableName = myFile.name.split('.')[0]
            df.to_sql(name=tableName, con=con, if_exists='replace', index=False, dtype=dtypedict)
            return redirect(reverse('show_table'))
        return redirect(reverse('show_table'))
class AddDatabaseView(View):
    def get(self, request):
        return render(request, 'addDatabase.html')
class ShowTableView(View):
    def get(self,request):
        # import pymysql
        # # 打开数据库连接
        # db = pymysql.connect("localhost", "root", "root123", "ai2")
        # # 使用 cursor() 方法创建一个游标对象 cursor
        # cursor = db.cursor(pymysql.cursors.DictCursor)
        # # 使用 execute()  方法执行 SQL 查询
        # sql = "show tables"
        # cursor.execute(sql)
        # # 使用 fetchone() 方法获取单条数据.
        # data = cursor.fetchall()
        # # 关闭数据库连接
        # db.close()
        import os
        filePath = './static/upload/'
        datas = os.listdir(filePath)
        data = [i.split('.')[0] for i in datas]
        return render(request,'show_table.html',{'tables':data})
class ConDataView(View):
    def get(self,request):
        return redirect(reverse('show_table'))
class SqlProcessView(View):
    def get(self,request):
        return render(request,'sql_process.html')

class TextProcessView(View):
    def get(self,request):
        import os
        filePath = './static/upload/'
        datas = os.listdir(filePath)
        l = []
        for i in datas:
            if i.split('.')[1] == 'txt':
                l.append(i.split('.')[0])
        return render(request,'text_process.html',{'tables':l})
class Model2View(View):
    def get(self, request):
        return render(request, 'model.html')
def dbUtils(sql):
    import pymysql
    # 打开数据库连接
    db = pymysql.connect("localhost", "root", "root123", "ai2")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor(pymysql.cursors.DictCursor)
    cursor.execute(sql)
    # 使用 fetchone() 方法获取单条数据.
    data = cursor.fetchall()
    # 关闭数据库连接
    db.commit()
    db.close()
    return data
class SqlExeView(View):
    def get(self, request):
        return render(request, 'sql_exe.html')
    def post(self, request):
        sql = request.POST.get('txt','').lower()
        sql = sql.replace("\r\n",'')
        table_name = sql.split('from ')[1]
        table_name = table_name.split(' ')[0]
        data = dbUtils(sql)
        print(data)
        paginator = Paginator(data,10)
        print(paginator)
        current_page_num = request.POST.get('page',1)
        table = paginator.page(current_page_num)
        return render(request, 'sql_exe.html',{'table':table,'table_name':table_name})

def kmeans(request):
    response = {'state':True}
    col_id = request.POST.get('col_id','')
    table_name = col_id.split(',')[0]
    col_name = col_id.split(',')[1]
    sql = "select " + col_name + " from " + table_name
    data = dbUtils(sql)
    data = [ [x[col_name],x[col_name]]  for x in data]
    data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1], marker='o')
    plt.show()
    y_pred = KMeans(n_clusters=2).fit_predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=y_pred)
    plt.xlabel('num')
    plt.ylabel('num')
    plt.title('outlier')
    plt.savefig('./static/images/yichang.png')
    plt.show()
    return JsonResponse(response,safe=False)
def table_edit(request,edit_id):
    if request.method == "GET":
        sql = 'select * from '
        table_name = request.GET.get('t_name','')
        sql = sql + table_name + ' where id = '+edit_id
        print(sql)
        data = dbUtils(sql)
        return render(request, "table_edit.html",{"table": data,'table_name':table_name})
    else:
        table_name = request.POST.get("t_name")
        id = request.POST.get('col_id')
        col_name = request.POST.get("col_name")
        col_value = request.POST.get("col_value")
        print(col_value)
        sql = 'update '+table_name+' set '+ col_name +' = "'+col_value+'" where id = '+ id
        import pymysql
        conn = pymysql.connect(user='root', passwd='root123',
                               host='localhost', db='ai2', charset='utf8')
        cur = conn.cursor()
        print(sql)
        sta = cur.execute(sql)
        if sta == 1:
            print('Done')
        else:
            print('Failed')
        conn.commit()
        cur.close()
        conn.close()
        return HttpResponse('更新完成')


def table_delete(request,delete_id):
    response = {'state': True}
    if request.method == "GET":
        table_name = request.GET.get("t_name",'')
        id = delete_id
        sql = 'DELETE FROM '+table_name+' WHERE ID = '+ id
        print(sql)
        import pymysql
        conn = pymysql.connect(user='root', passwd='root123',
                               host='localhost', db='ai2', charset='utf8')
        cur = conn.cursor()
        print(sql)
        sta = cur.execute(sql)
        if sta == 1:
            print('Done')
        else:
            print('Failed')
        conn.commit()
        cur.close()
        conn.close()
    return JsonResponse(response,safe=False)