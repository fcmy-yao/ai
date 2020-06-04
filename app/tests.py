from app.DataPreprocessing_add_category import data


class RunView(View):
    def get(self,request):
        train_check_num = []
        import sys
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import MinMaxScaler
        from collections import Counter
        import random
        plt.rcParams['font.sans-serif'] = ['SimHei']
        MinMax = MinMaxScaler()
        np.set_printoptions(suppress=True, threshold=sys.maxsize)
        data = pd.read_csv('C:/Users/yql/PycharmProjects/ai/app/yunfu_visit_neonate12.csv')

        # 下列特征依次为：[id,checkweek,chushengyz,age,height,weight,heightfundusuterus,abdomencircumference,
        # dbp,sbp,fetalheartrate,bloodtypecode,rhbloodcode,maritalstatuscode,gravidity,16-parity,
        # menarcheage,menstrualperiod,cycle,menstrualblood,dysmenorrhea,trafficflow,naturalabortion,
        # odinopoeia,preterm,dystocia,died,abnormality,newbrondied,qwetimes,ectopicpregnancy,
        # vesicularmole,highriskreason]
        # features1=data.iloc[:,[0,3,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,
        #                        28,29,30,31,32,33,34,35,36,37,40]]

        features1 = data.iloc[:, [0, 7, 3, 8, 9, 10, 11, 12, 20, 23, 24, 27, 30, 40]]
        # 0-id，7-出生孕周，3-体检周，8-年龄，9-身高，10-体重，11-宫高，12-腹围，20-胎次，23-经期时间长，24-月经周期，27-人流史，30-早产史，40-高危
        # 序号: 0-id, 1-出生孕周, 2-体检周, 3-年龄, 4-身高, 5-体重, 6-宫高, 7-腹围, 8-胎次, 9-经期时间长, 10-月经周期, 11-人流史, 12-早产史, 13-高危
        # features1=data.iloc[:,[0,7,3,8,9,10,11,12]]

        labels1 = data.iloc[:, [0, 44]]

        features1 = np.array(features1)
        labels1 = np.array(labels1)
        print(features1.shape)
        features2 = np.zeros([len(features1), 54])
        # multi_hot编码，并拼接到原来的特征后面
        for i in range(len(features1)):
            multi_hot = np.zeros([41])
            x = np.fromstring(features1[i][13], int, sep=';')
            for j in range(len(x)):
                if x[j] < 42:
                    multi_hot[(x[j] - 1)] = 1
            features2[i] = np.concatenate((features1[i][0:13], multi_hot), axis=0)

        print(features2.shape)
        # 胎盘异常，胎位不正，胎膜早破，糖尿病，辅助生殖，多胎
        # 10，     11，     13，     21，   27，     39      对应高危表编号

        features3 = features2[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 22, 23, 25, 33, 39, 51]]
        # 0-id, 1-出生孕周, 2-体检周, 3-年龄, 4-身高, 5-体重, 6-宫高, 7-腹围, 8-胎次, 12-早产史,
        # 22-胎盘异常, 23-胎位不正, 25-胎膜早破, 33-糖尿病, 39-辅助生殖, 51-多胎

        # 序号: 0-id, 1-出生孕周, 2-体检周, 3-年龄, 4-身高, 5-体重, 6-宫高, 7-腹围, 8-胎次, 9-早产史,
        # 10-胎盘异常, 11-胎位不正, 12-胎膜早破, 13-糖尿病, 14-辅助生殖, 15-多胎

        print(features3.shape)

        # 找到NAN
        for i in range(len(features3)):
            for j in range(len(features3[i])):
                if np.isnan(features3[i][j]):
                    features3[i][j] = 0

        X_dict = {}
        Y_dict = {}
        checkweek = {}

        # 特征入字典
        # 第0列是id,第2列是checkweek
        for i in range(len(features3)):
            if features3[i, 0] not in X_dict:
                X_dict[features3[i, 0]] = []
                checkweek[features3[i, 0]] = []
                X_dict[features3[i, 0]].append(np.array(features3[i, 1:]))
                checkweek[features3[i, 0]].append(np.array(features3[i, 2]))

            elif ((features3[i, 0] in X_dict) and (features3[i, 2] not in checkweek[features3[i, 0]])):
                X_dict[features3[i, 0]].append(np.array(features3[i, 1:]))
                checkweek[features3[i, 0]].append(features3[i, 2])
        ret_list = []
        res_list = []
        res_list.append('孕妇总数:'+ str(len(X_dict)))
        print('孕妇总数:', len(X_dict))

        # 标签入字典
        for j in range(len(labels1)):
            if labels1[j, 0] not in Y_dict:
                Y_dict[labels1[j, 0]] = labels1[j, 1]  # 第一个标签

        # 找出最大体检次数
        max_length = 0
        for i in X_dict.keys():
            if len(X_dict[i]) > max_length:
                max_length = len(X_dict[i])
        print('最大体检序列长度为:', max_length)
        # 填充时间步
        for i in X_dict.keys():
            while len(X_dict[i]) < max_length:
                X_dict[i].append(np.array(np.zeros(features3.shape[1] - 1)))

        # 字典转列表，自然去除id
        X_list = []
        Y_list = []
        for i in X_dict.keys():  # 即可以升序的方式遍历X_dict中的key
            X_list.append(X_dict[i])
            Y_list.append(Y_dict[i])

        # 列表转矩阵
        X_matrix = np.array(X_list)
        Y_matrix = np.array(Y_list)

        # 筛选最后一次体检和分娩在同一周的孕妇
        select_X_list = []
        select_Y_list = []
        check_num = []
        for i in range(len(X_matrix)):
            count = 0
            for j in range(len(X_matrix[i])):
                if sum(X_matrix[i, j, :]) != 0.0:
                    count += 1
            check_num.append(count)

        for i in range(len(X_matrix)):
            if X_matrix[i, check_num[i] - 1, 0] == X_matrix[i, check_num[i] - 1, 1]:
                select_X_list.append(X_matrix[i])
                select_Y_list.append(Y_matrix[i])

        X_matrix = np.array(select_X_list)[:, :, 1:]
        Y_matrix = np.array(select_Y_list)

        print('总特征维度:', X_matrix.shape)
        print('总标签维度:', Y_matrix.shape)
        ret_list.append('总特征维度:' + str( X_matrix.shape))
        ret_list.append('总标签维度:' + str( Y_matrix.shape))
        X_features = X_matrix
        Y_labels = Y_matrix

        # 定义计算数据中低体重、正常、巨大儿的数量的函数
        def weight_category(labels):
            # 把各类样本抽出来
            Y_low_list = []
            Y_middle_list = []
            Y_high_list = []

            for i in range(len(labels)):
                if labels[i] < 2500:
                    Y_low_list.append(labels[i])
                if labels[i] >= 2500 and labels[i] <= 4000:
                    Y_middle_list.append(labels[i])
                if labels[i] > 4000:
                    Y_high_list.append(labels[i])
            print('低体重数量:', len(Y_low_list))
            print('正常体重数量:', len(Y_middle_list))
            print('巨大儿数量:', len(Y_high_list))
            print('')

        # 体检周归一化，选择[11, ]大于等于11的周,先去除异常值，后拆矩阵，不做分段缩放---------------------------------------------
        X_Temp = []
        Y_Temp = []
        for i in range(len(X_features)):
            #    for j in range(len(X_features[i])):
            if X_features[i, 0, 0] >= 11:
                X_Temp.append(X_features[i])
                Y_Temp.append(Y_labels[i])
        X_features = np.array(X_Temp)
        Y_labels = np.array(Y_Temp)

        print('体检周去除异常值后:')
        weight_category(Y_labels)

        # 找出每个人的体检次数,用于还原矩阵
        check_num = []
        for i in range(len(X_features)):
            count = 0
            for j in range(len(X_features[i])):
                if sum(X_features[i, j, :]) != 0:
                    count += 1
            check_num.append(count)

        # 拆矩阵
        X_Temp_checkweek = []
        for i in range(len(X_features)):
            for j in range(len(X_features[i])):
                if sum(X_features[i, j, :]) != 0:
                    X_Temp_checkweek.append(X_features[i, j, 0])  # 选取体检周特征
        X_Temp_checkweek = list(np.squeeze(MinMax.fit_transform(np.array(X_Temp_checkweek).reshape(-1, 1))))

        # 还原矩阵
        matrix = []
        checkweek_count = 0
        for i in range(len(check_num)):
            matrix.append(list(X_Temp_checkweek[checkweek_count:check_num[i] + checkweek_count]))
            checkweek_count += check_num[i]
        for i in range(len(matrix)):
            while len(matrix[i]) < max_length:
                matrix[i].append(0.0)
        X_features = np.concatenate((np.array(X_features), np.array(matrix).reshape(np.array(matrix).shape[0],
                                                                                    np.array(matrix).shape[1],
                                                                                    1)), axis=2)
        # print('X_features:',X_features.shape)

        # 体检周处理完毕--------------------------------------------------------------------------------------
        # 仍保持特征和标签放在 X_features,Y_labels 这两个变量中

        # 年龄归一化，选择[17, 45]的范围,不做分段缩放------------------------------------------------------------
        X_Temp = []
        Y_Temp = []
        for i in range(len(X_features)):
            if X_features[i, 0, 1] >= 17 and X_features[i, 0, 1] <= 45:
                X_Temp.append(X_features[i])
                Y_Temp.append(Y_labels[i])

        X_features = np.array(X_Temp)
        Y_labels = np.array(Y_Temp)
        print('年龄去除异常值后:')
        weight_category(Y_labels)
        # print('年龄限定范围后:',X_features.shape)
        # print('年龄限定范围后:',Y_labels.shape)

        # 找出每个人的体检次数,用于还原矩阵
        check_num = []
        for i in range(len(X_features)):
            count = 0
            for j in range(len(X_features[i])):
                if sum(X_features[i, j, :]) != 0:
                    count += 1
            check_num.append(count)

        # 拆矩阵
        X_Temp_age = []
        for i in range(len(X_features)):
            for j in range(len(X_features[i])):
                if sum(X_features[i, j, :]) != 0:
                    X_Temp_age.append(X_features[i, j, 1])  # 选取年龄特征
        X_Temp_age = list(np.squeeze(MinMax.fit_transform(np.array(X_Temp_age).reshape(-1, 1))))

        # 还原矩阵
        matrix = []
        age_count = 0
        for i in range(len(check_num)):
            matrix.append(list(X_Temp_age[age_count:check_num[i] + age_count]))
            age_count += check_num[i]
        for i in range(len(matrix)):
            while len(matrix[i]) < max_length:
                matrix[i].append(0.0)
        X_features = np.concatenate((np.array(X_features), np.array(matrix).reshape(np.array(matrix).shape[0],
                                                                                    np.array(matrix).shape[1],
                                                                                    1)), axis=2)
        # print('X_features:',X_features.shape)
        # 年龄特征处理完毕--------------------------------------------------------------------------------------
        # 仍保持特征和标签放在 X_features,Y_labels 这两个变量中

        # 身高取范围[142, 180],分段缩放[142, 151], [151, 174], [174, 180]--------------------------------------
        X_Temp = []
        Y_Temp = []
        for i in range(len(X_features)):
            if X_features[i, 0, 2] >= 142 and X_features[i, 0, 2] <= 180:
                X_Temp.append(X_features[i])
                Y_Temp.append(Y_labels[i])
        X_features = np.array(X_Temp)
        Y_labels = np.array(Y_Temp)
        print('身高去除异常值后:')
        weight_category(Y_labels)
        # print('身高限定范围后:',X_features.shape)
        # print('身高限定范围后:',Y_labels.shape)

        x1, x2, x3 = [], [], []
        y1, y2, y3 = [], [], []
        for i in range(len(X_features)):
            if X_features[i, 0, 2] >= 142 and X_features[i, 0, 2] <= 151:
                x1.append(X_features[i])
                y1.append(Y_labels[i])
            if X_features[i, 0, 2] > 151 and X_features[i, 0, 2] <= 174:
                x2.append(X_features[i])
                y2.append(Y_labels[i])
            if X_features[i, 0, 2] > 174 and X_features[i, 0, 2] <= 180:
                x3.append(X_features[i])
                y3.append(Y_labels[i])

        height_len_interval1 = 151 - 142
        height_threshold_interval1 = 142

        height_len_interval2 = 174 - 151
        height_threshold_interval2 = 151

        height_len_interval3 = 180 - 174
        height_threshold_interval3 = 174

        X_features = np.array(x1 + x2 + x3)
        Y_labels = np.array(y1 + y2 + y3)

        # print(X_features.shape)
        # print(Y_labels.shape)

        x1, x2, x3 = np.array(x1), np.array(x2), np.array(x3)

        # 分别对每段拆矩阵,处理x1段
        # 找出每个人的体检次数,用于还原矩阵
        x1_check_num = []
        for i in range(len(x1)):
            count = 0
            for j in range(len(x1[i])):
                if sum(x1[i, j, :]) != 0:
                    count += 1
            x1_check_num.append(count)

        # 拆矩阵
        x1_Temp_height = []
        for i in range(len(x1)):
            for j in range(len(x1[i])):  # 此处多做的计算是为了方便还原矩阵
                if sum(x1[i, j, :]) != 0:
                    x1_Temp_height.append(x1[i, j, 2])  # 选取身高特征

        # 归一化
        # 区间密度=不同的数量/区间长度
        height_x1_D = int(500 * len(dict(Counter(x1_Temp_height)).keys()) / (max(x1_Temp_height) - min(x1_Temp_height)))
        x1_Temp_height = list(
            np.squeeze(height_x1_D * (np.array(x1_Temp_height) - height_threshold_interval1) / height_len_interval1))

        # 处理x2段
        # 找出每个人的体检次数,用于还原矩阵
        x2_check_num = []
        for i in range(len(x2)):
            count = 0
            for j in range(len(x2[i])):
                if sum(x2[i, j, :]) != 0:
                    count += 1
            x2_check_num.append(count)

        # 拆矩阵
        x2_Temp_height = []
        for i in range(len(x2)):
            for j in range(len(x2[i])):  # 此处多做的计算是为了方便还原矩阵
                if sum(x2[i, j, :]) != 0:
                    x2_Temp_height.append(x2[i, j, 2])  # 选取身高特征
        # 归一化
        # 区间密度=不同的数量/区间长度
        height_x2_D = int(500 * len(dict(Counter(x2_Temp_height)).keys()) / (max(x2_Temp_height) - min(x2_Temp_height)))
        x2_Temp_height = list(np.squeeze(height_x2_D * (
                    (np.array(x2_Temp_height) - height_threshold_interval2) / height_len_interval2) + height_x1_D))

        # 处理x3段
        # 找出每个人的体检次数,用于还原矩阵
        x3_check_num = []
        for i in range(len(x3)):
            count = 0
            for j in range(len(x3[i])):
                if sum(x3[i, j, :]) != 0:
                    count += 1
            x3_check_num.append(count)

        # 拆矩阵
        x3_Temp_height = []
        for i in range(len(x3)):
            for j in range(len(x3[i])):
                if sum(x3[i, j, :]) != 0:
                    x3_Temp_height.append(x3[i, j, 2])  # 选取身高特征

        # 一次归一化
        # 区间密度=不同的数量/区间长度
        height_x3_D = int(500 * len(dict(Counter(x3_Temp_height)).keys()) / (max(x3_Temp_height) - min(x3_Temp_height)))
        x3_Temp_height = list(np.squeeze(height_x3_D * ((np.array(
            x3_Temp_height) - height_threshold_interval3) / height_len_interval3) + height_x1_D + height_x2_D))

        # 二次归一化
        sum_Temp = MinMax.fit_transform(np.array(x1_Temp_height + x2_Temp_height + x3_Temp_height).reshape(-1, 1))
        x1_Temp_height = list(sum_Temp[:len(x1_Temp_height)])
        x2_Temp_height = list(sum_Temp[len(x1_Temp_height):len(x1_Temp_height) + len(x2_Temp_height)])
        x3_Temp_height = list(sum_Temp[
                              len(x1_Temp_height) + len(x2_Temp_height):len(x1_Temp_height) + len(x2_Temp_height) + len(
                                  x3_Temp_height)])

        # 还原矩阵
        x1_matrix = []
        x1_height_count = 0
        for i in range(len(x1_check_num)):
            x1_matrix.append(list(x1_Temp_height[x1_height_count:x1_check_num[i] + x1_height_count]))
            x1_height_count += x1_check_num[i]
        for i in range(len(x1_matrix)):
            while len(x1_matrix[i]) < max_length:
                x1_matrix[i].append(0.0)
        x1_matrix = np.array(x1_matrix)

        # 还原矩阵
        x2_matrix = []
        x2_height_count = 0
        for i in range(len(x2_check_num)):
            x2_matrix.append(list(x2_Temp_height[x2_height_count:x2_check_num[i] + x2_height_count]))
            x2_height_count += x2_check_num[i]
        for i in range(len(x2_matrix)):
            while len(x2_matrix[i]) < max_length:
                x2_matrix[i].append(0.0)
        x2_matrix = np.array(x2_matrix)

        # 还原矩阵
        x3_matrix = []
        x3_height_count = 0
        for i in range(len(x3_check_num)):
            x3_matrix.append(list(x3_Temp_height[x3_height_count:x3_check_num[i] + x3_height_count]))
            x3_height_count += x3_check_num[i]
        for i in range(len(x3_matrix)):
            while len(x3_matrix[i]) < max_length:
                x3_matrix[i].append(0.0)
        x3_matrix = np.array(x3_matrix)

        # print(np.expand_dims(np.concatenate((x1_matrix, x2_matrix, x3_matrix), axis=0),axis=2).shape)
        X_features = np.concatenate(
            (X_features, np.expand_dims(np.concatenate((x1_matrix, x2_matrix, x3_matrix), axis=0), axis=2)), axis=2)

        # print(X_features.shape)
        # print(Y_labels.shape)
        print('身高归一化后:')
        weight_category(Y_labels)
        # 身高归一化完毕-------------------------------------------------------------------------

        # 体重取范围[36, 93],分段缩放[36, 44], [44, 76], [76, 82], [82, 93]
        X_Temp = []
        Y_Temp = []
        for i in range(len(X_features)):
            scope = []
            for j in range(len(X_features[i])):
                if X_features[i, j, 3] >= 36 and X_features[i, j, 3] <= 93 or X_features[i, j, 3] == 0.0:
                    scope.append(X_features[i, j, 3])
            if len(scope) == X_features.shape[1]:
                X_Temp.append(X_features[i])
                Y_Temp.append(Y_labels[i])

        X_features = np.array(X_Temp)
        Y_labels = np.array(Y_Temp)
        # print('体重限定范围后:',X_features.shape)
        # print('体重限定范围后:',Y_labels.shape)
        print('体重去除异常值后:')
        weight_category(Y_labels)

        # 找出每个人的体检次数,用于还原矩阵
        weight_check_num = []
        for i in range(len(X_features)):
            count = 0
            for j in range(len(X_features[i])):
                if sum(X_features[i, j, :]) != 0:
                    count += 1
            weight_check_num.append(count)

        Temp_weight_raw = []
        x1_Temp_weight, x2_Temp_weight, x3_Temp_weight, x4_Temp_weight = [], [], [], []
        x1_Temp_weight_index, x2_Temp_weight_index, x3_Temp_weight_index, x4_Temp_weight_index = [], [], [], []

        for i in range(len(X_features)):
            for j in range(len(X_features[i])):
                if X_features[i, j, 3] != 0.0:
                    Temp_weight_raw.append(X_features[i, j, 3])

        Temp_weight_new = np.zeros(len(Temp_weight_raw))

        for i in range(len(Temp_weight_raw)):
            if Temp_weight_raw[i] >= 36 and Temp_weight_raw[i] <= 44:
                x1_Temp_weight.append(Temp_weight_raw[i])
                x1_Temp_weight_index.append(i)  # 保存索引，用于还原列表
            if Temp_weight_raw[i] > 44 and Temp_weight_raw[i] <= 76:
                x2_Temp_weight.append(Temp_weight_raw[i])
                x2_Temp_weight_index.append(i)
            if Temp_weight_raw[i] > 76 and Temp_weight_raw[i] <= 82:
                x3_Temp_weight.append(Temp_weight_raw[i])
                x3_Temp_weight_index.append(i)
            if Temp_weight_raw[i] > 82 and Temp_weight_raw[i] <= 93:
                x4_Temp_weight.append(Temp_weight_raw[i])
                x4_Temp_weight_index.append(i)

        weight_len_interval1 = 44 - 36
        weight_threshold_interval1 = 36

        weight_len_interval2 = 76 - 44
        weight_threshold_interval2 = 44

        weight_len_interval3 = 82 - 76
        weight_threshold_interval3 = 76

        weight_len_interval4 = 93 - 82
        weight_threshold_interval4 = 82

        # 一次归一化
        weight_x1_D = int(500 * len(dict(Counter(x1_Temp_weight)).keys()) / (max(x1_Temp_weight) - min(x1_Temp_weight)))
        weight_x2_D = int(500 * len(dict(Counter(x2_Temp_weight)).keys()) / (max(x2_Temp_weight) - min(x2_Temp_weight)))
        weight_x3_D = int(500 * len(dict(Counter(x3_Temp_weight)).keys()) / (max(x3_Temp_weight) - min(x3_Temp_weight)))
        weight_x4_D = int(500 * len(dict(Counter(x4_Temp_weight)).keys()) / (max(x4_Temp_weight) - min(x4_Temp_weight)))

        x1_Temp_weight = list(
            np.squeeze(weight_x1_D * ((np.array(x1_Temp_weight) - weight_threshold_interval1) / weight_len_interval1)))
        x2_Temp_weight = list(np.squeeze(weight_x2_D * (
                    (np.array(x2_Temp_weight) - weight_threshold_interval2) / weight_len_interval2) + weight_x1_D))
        x3_Temp_weight = list(np.squeeze(weight_x3_D * ((np.array(
            x3_Temp_weight) - weight_threshold_interval3) / weight_len_interval3) + weight_x1_D + weight_x2_D))
        x4_Temp_weight = list(np.squeeze(weight_x4_D * ((np.array(
            x4_Temp_weight) - weight_threshold_interval4) / weight_len_interval4) + weight_x1_D + weight_x2_D + weight_x3_D))

        # 二次归一化
        sum_Temp_weight = MinMax.fit_transform(
            np.array(x1_Temp_weight + x2_Temp_weight + x3_Temp_weight + x4_Temp_weight).reshape(-1, 1))
        x1_Temp_weight = list(sum_Temp_weight[:len(x1_Temp_weight)])
        x2_Temp_weight = list(sum_Temp_weight[len(x1_Temp_weight):len(x1_Temp_weight) + len(x2_Temp_weight)])
        x3_Temp_weight = list(sum_Temp_weight[
                              len(x1_Temp_weight) + len(x2_Temp_weight):len(x1_Temp_weight) + len(x2_Temp_weight) + len(
                                  x3_Temp_weight)])
        x4_Temp_weight = list(sum_Temp_weight[
                              len(x1_Temp_weight) + len(x2_Temp_weight) + len(x3_Temp_weight):len(x1_Temp_weight) + len(
                                  x2_Temp_weight) + len(x3_Temp_weight) + len(x4_Temp_weight)])

        # 还原
        x_weight = x1_Temp_weight + x2_Temp_weight + x3_Temp_weight + x4_Temp_weight
        index_weight = x1_Temp_weight_index + x2_Temp_weight_index + x3_Temp_weight_index + x4_Temp_weight_index
        for i in range(len(x_weight)):
            Temp_weight_new[index_weight[i]] = x_weight[i]

        # 还原矩阵
        weight_matrix = []
        weight_count = 0
        for i in range(len(weight_check_num)):
            weight_matrix.append(list(Temp_weight_new[weight_count:weight_check_num[i] + weight_count]))
            weight_count += weight_check_num[i]
        for i in range(len(weight_matrix)):
            while len(weight_matrix[i]) < max_length:
                weight_matrix[i].append(0.0)
        weight_matrix = np.array(weight_matrix)
        X_features = np.concatenate((X_features, np.expand_dims(weight_matrix, axis=2)), axis=2)

        # print(X_features.shape)
        print('体重归一化后:')
        weight_category(Y_labels)
        # 体重归一化完毕---------------------------------------------------------------------------------

        # 宫高取范围[7, 48],分段缩放[7, 13.5],[13.5, 37], [37, 48]
        X_Temp = []
        Y_Temp = []
        for i in range(len(X_features)):
            scope = []
            for j in range(len(X_features[i])):
                if (X_features[i, j, 4] >= 7 and X_features[i, j, 4] <= 48) or X_features[i, j, 4] == 0.0:
                    scope.append(X_features[i, j, 4])
            if len(scope) == X_features.shape[1]:
                X_Temp.append(X_features[i])
                Y_Temp.append(Y_labels[i])

        X_features = np.array(X_Temp)
        Y_labels = np.array(Y_Temp)
        # print('宫高限定范围后:',X_features.shape)
        # print('宫高限定范围后:',Y_labels.shape)
        print('宫高去除异常值后:')
        weight_category(Y_labels)

        # 找出每个人的体检次数,用于还原矩阵
        heightfundusuterus_check_num = []
        for i in range(len(X_features)):
            count = 0
            for j in range(len(X_features[i])):
                if sum(X_features[i, j, :]) != 0:
                    count += 1
            heightfundusuterus_check_num.append(count)

        Temp_heightfundusuterus_raw = []
        x1_Temp_heightfundusuterus, x2_Temp_heightfundusuterus, x3_Temp_heightfundusuterus = [], [], []
        x1_Temp_heightfundusuterus_index, x2_Temp_heightfundusuterus_index, x3_Temp_heightfundusuterus_index = [], [], []

        for i in range(len(X_features)):
            for j in range(len(X_features[i])):
                if X_features[i, j, 4] != 0.0:
                    Temp_heightfundusuterus_raw.append(X_features[i, j, 4])

        Temp_heightfundusuterus_new = np.zeros(len(Temp_heightfundusuterus_raw))

        for i in range(len(Temp_heightfundusuterus_raw)):
            if Temp_heightfundusuterus_raw[i] >= 7 and Temp_heightfundusuterus_raw[i] <= 13.5:
                x1_Temp_heightfundusuterus.append(Temp_heightfundusuterus_raw[i])
                x1_Temp_heightfundusuterus_index.append(i)  # 保存索引，用于还原列表
            if Temp_heightfundusuterus_raw[i] > 13.5 and Temp_heightfundusuterus_raw[i] <= 37:
                x2_Temp_heightfundusuterus.append(Temp_heightfundusuterus_raw[i])
                x2_Temp_heightfundusuterus_index.append(i)
            if Temp_heightfundusuterus_raw[i] > 37 and Temp_heightfundusuterus_raw[i] <= 48:
                x3_Temp_heightfundusuterus.append(Temp_heightfundusuterus_raw[i])
                x3_Temp_heightfundusuterus_index.append(i)

        heightfundusuterus_len_interval1 = 13.5 - 7
        heightfundusuterus_threshold_interval1 = 7

        heightfundusuterus_len_interval2 = 37 - 13.5
        heightfundusuterus_threshold_interval2 = 13.5

        heightfundusuterus_len_interval3 = 48 - 37
        heightfundusuterus_threshold_interval3 = 37

        # 一次归一化
        heightfundusuterus_x1_D = int(500 * len(dict(Counter(x1_Temp_heightfundusuterus)).keys()) / (
                    max(x1_Temp_heightfundusuterus) - min(x1_Temp_heightfundusuterus)))
        heightfundusuterus_x2_D = int(500 * len(dict(Counter(x2_Temp_heightfundusuterus)).keys()) / (
                    max(x2_Temp_heightfundusuterus) - min(x2_Temp_heightfundusuterus)))
        heightfundusuterus_x3_D = int(500 * len(dict(Counter(x3_Temp_heightfundusuterus)).keys()) / (
                    max(x3_Temp_heightfundusuterus) - min(x3_Temp_heightfundusuterus)))

        x1_Temp_heightfundusuterus = list(np.squeeze(heightfundusuterus_x1_D * ((np.array(
            x1_Temp_heightfundusuterus) - heightfundusuterus_threshold_interval1) / heightfundusuterus_len_interval1)))
        x2_Temp_heightfundusuterus = list(np.squeeze(heightfundusuterus_x2_D * ((np.array(
            x2_Temp_heightfundusuterus) - heightfundusuterus_threshold_interval2) / heightfundusuterus_len_interval2) + heightfundusuterus_x1_D))
        x3_Temp_heightfundusuterus = list(np.squeeze(heightfundusuterus_x3_D * ((np.array(
            x3_Temp_heightfundusuterus) - heightfundusuterus_threshold_interval3) / heightfundusuterus_len_interval3) + heightfundusuterus_x1_D + heightfundusuterus_x2_D))

        # 二次归一化
        sum_Temp_heightfundusuterus = MinMax.fit_transform(
            np.array(x1_Temp_heightfundusuterus + x2_Temp_heightfundusuterus + x3_Temp_heightfundusuterus).reshape(-1,
                                                                                                                   1))
        x1_Temp_heightfundusuterus = list(sum_Temp_heightfundusuterus[:len(x1_Temp_heightfundusuterus)])
        x2_Temp_heightfundusuterus = list(sum_Temp_heightfundusuterus[
                                          len(x1_Temp_heightfundusuterus):len(x1_Temp_heightfundusuterus) + len(
                                              x2_Temp_heightfundusuterus)])
        x3_Temp_heightfundusuterus = list(sum_Temp_heightfundusuterus[
                                          len(x1_Temp_heightfundusuterus) + len(x2_Temp_heightfundusuterus):len(
                                              x1_Temp_heightfundusuterus) + len(x2_Temp_heightfundusuterus) + len(
                                              x3_Temp_heightfundusuterus)])

        # 还原
        x_heightfundusuterus = x1_Temp_heightfundusuterus + x2_Temp_heightfundusuterus + x3_Temp_heightfundusuterus
        index_heightfundusuterus = x1_Temp_heightfundusuterus_index + x2_Temp_heightfundusuterus_index + x3_Temp_heightfundusuterus_index
        for i in range(len(x_heightfundusuterus)):
            Temp_heightfundusuterus_new[index_heightfundusuterus[i]] = x_heightfundusuterus[i]

        # 还原矩阵
        heightfundusuterus_matrix = []
        heightfundusuterus_count = 0
        for i in range(len(heightfundusuterus_check_num)):
            heightfundusuterus_matrix.append(list(Temp_heightfundusuterus_new[
                                                  heightfundusuterus_count:heightfundusuterus_check_num[
                                                                               i] + heightfundusuterus_count]))
            heightfundusuterus_count += heightfundusuterus_check_num[i]
        for i in range(len(heightfundusuterus_matrix)):
            while len(heightfundusuterus_matrix[i]) < max_length:
                heightfundusuterus_matrix[i].append(0.0)
        heightfundusuterus_matrix = np.array(heightfundusuterus_matrix)
        X_features = np.concatenate((X_features, np.expand_dims(heightfundusuterus_matrix, axis=2)), axis=2)

        # print(X_features.shape)
        print('宫高归一化后:')
        weight_category(Y_labels)
        # 宫高归一化完毕---------------------------------------------------------------------------------

        # 腹围未限定范围,分段缩放(*, 63], [63, 119], [119, *)
        '''
        X_Temp=[]
        Y_Temp=[]

        for i in range(len(X_features)):
            scope=[]
            for j in range(len(X_features[i])):
                if (X_features[i,j,4]>=6 and X_features[i,j,4]<=48) or X_features[i,j,4]==0.0:
                    scope.append(X_features[i,j,4])
            if len(scope)==X_features.shape[1]:
                X_Temp.append(X_features[i])
                Y_Temp.append(Y_labels[i])

        X_features=np.array(X_Temp)
        Y_labels=np.array(Y_Temp)
        '''
        # print('腹围限定范围后:',X_features.shape)
        # print('腹围限定范围后:',Y_labels.shape)
        print('腹围不必限定范围')

        # 找出每个人的体检次数,用于还原矩阵
        abdom_check_num = []
        for i in range(len(X_features)):
            count = 0
            for j in range(len(X_features[i])):
                if sum(X_features[i, j, :]) != 0:
                    count += 1
            abdom_check_num.append(count)

        Temp_abdom_raw = []
        x1_Temp_abdom, x2_Temp_abdom, x3_Temp_abdom = [], [], []
        x1_Temp_abdom_index, x2_Temp_abdom_index, x3_Temp_abdom_index = [], [], []

        for i in range(len(X_features)):
            for j in range(len(X_features[i])):
                if X_features[i, j, 5] != 0.0:
                    Temp_abdom_raw.append(X_features[i, j, 5])

        Temp_abdom_new = np.zeros(len(Temp_abdom_raw))

        for i in range(len(Temp_abdom_raw)):
            if Temp_abdom_raw[i] <= 63:
                x1_Temp_abdom.append(Temp_abdom_raw[i])
                x1_Temp_abdom_index.append(i)  # 保存索引，用于还原列表
            if Temp_abdom_raw[i] > 63 and Temp_abdom_raw[i] <= 119:
                x2_Temp_abdom.append(Temp_abdom_raw[i])
                x2_Temp_abdom_index.append(i)
            if Temp_abdom_raw[i] > 119:
                x3_Temp_abdom.append(Temp_abdom_raw[i])
                x3_Temp_abdom_index.append(i)

        abdom_len_interval1 = 63 - min(x1_Temp_abdom)  # 定位边界的区间跨度
        abdom_threshold_interval1 = min(x1_Temp_abdom)

        abdom_len_interval2 = 119 - 63
        abdom_threshold_interval2 = 63

        abdom_len_interval3 = max(x3_Temp_abdom) - 119
        abdom_threshold_interval3 = 119

        # 一次归一化
        abdom_x1_D = int(500 * len(dict(Counter(x1_Temp_abdom)).keys()) / (max(x1_Temp_abdom) - min(x1_Temp_abdom)))
        abdom_x2_D = int(500 * len(dict(Counter(x2_Temp_abdom)).keys()) / (max(x2_Temp_abdom) - min(x2_Temp_abdom)))
        abdom_x3_D = int(500 * len(dict(Counter(x3_Temp_abdom)).keys()) / (max(x3_Temp_abdom) - min(x3_Temp_abdom)))

        x1_Temp_abdom = list(
            np.squeeze(abdom_x1_D * ((np.array(x1_Temp_abdom) - abdom_threshold_interval1) / abdom_len_interval1)))
        x2_Temp_abdom = list(np.squeeze(
            abdom_x2_D * ((np.array(x2_Temp_abdom) - abdom_threshold_interval2) / abdom_len_interval2) + abdom_x1_D))
        x3_Temp_abdom = list(np.squeeze(abdom_x3_D * ((np.array(
            x3_Temp_abdom) - abdom_threshold_interval3) / abdom_len_interval3) + abdom_x1_D + abdom_x2_D))

        # 二次归一化
        sum_Temp_abdom = MinMax.fit_transform(np.array(x1_Temp_abdom + x2_Temp_abdom + x3_Temp_abdom).reshape(-1, 1))
        x1_Temp_abdom = list(sum_Temp_abdom[:len(x1_Temp_abdom)])
        x2_Temp_abdom = list(sum_Temp_abdom[len(x1_Temp_abdom):len(x1_Temp_abdom) + len(x2_Temp_abdom)])
        x3_Temp_abdom = list(sum_Temp_abdom[
                             len(x1_Temp_abdom) + len(x2_Temp_abdom):len(x1_Temp_abdom) + len(x2_Temp_abdom) + len(
                                 x3_Temp_abdom)])

        # 还原
        x_abdom = x1_Temp_abdom + x2_Temp_abdom + x3_Temp_abdom
        index_abdom = x1_Temp_abdom_index + x2_Temp_abdom_index + x3_Temp_abdom_index
        for i in range(len(x_abdom)):
            Temp_abdom_new[index_abdom[i]] = x_abdom[i]

        # 还原矩阵
        abdom_matrix = []
        abdom_count = 0
        for i in range(len(abdom_check_num)):
            abdom_matrix.append(list(Temp_abdom_new[abdom_count:abdom_check_num[i] + abdom_count]))
            abdom_count += abdom_check_num[i]
        for i in range(len(abdom_matrix)):
            while len(abdom_matrix[i]) < max_length:
                abdom_matrix[i].append(0.0)
        abdom_matrix = np.array(abdom_matrix)
        X_features = np.concatenate((X_features, np.expand_dims(abdom_matrix, axis=2)), axis=2)

        # print(X_features.shape)
        print('腹围归一化后:')
        weight_category(Y_labels)
        # 腹围归一化完毕---------------------------------------------------------------------------------

        # 胎次归一化，选择[0, 2]的范围,不做分段缩放------------------------------------------------------------
        X_Temp = []
        Y_Temp = []
        for i in range(len(X_features)):
            if X_features[i, 0, 6] >= 0 and X_features[i, 0, 6] <= 2:
                X_Temp.append(X_features[i])
                Y_Temp.append(Y_labels[i])

        X_features = np.array(X_Temp)
        Y_labels = np.array(Y_Temp)
        print('胎次去除异常值后:')
        weight_category(Y_labels)

        # 找出每个人的体检次数,用于还原矩阵
        check_num = []
        for i in range(len(X_features)):
            count = 0
            for j in range(len(X_features[i])):
                if sum(X_features[i, j, :]) != 0:
                    count += 1
            check_num.append(count)

        # 拆矩阵
        X_Temp_parity = []
        for i in range(len(X_features)):
            for j in range(len(X_features[i])):
                if sum(X_features[i, j, :]) != 0:
                    X_Temp_parity.append(X_features[i, j, 6])  # 选取胎次特征
        X_Temp_parity = list(np.squeeze(MinMax.fit_transform(np.array(X_Temp_parity).reshape(-1, 1))))

        # 还原矩阵
        matrix = []
        parity_count = 0
        for i in range(len(check_num)):
            matrix.append(list(X_Temp_parity[parity_count:check_num[i] + parity_count]))
            parity_count += check_num[i]
        for i in range(len(matrix)):
            while len(matrix[i]) < max_length:
                matrix[i].append(0.0)
        X_features = np.concatenate((np.array(X_features), np.array(matrix).reshape(np.array(matrix).shape[0],
                                                                                    np.array(matrix).shape[1],
                                                                                    1)), axis=2)
        # 胎次特征处理完毕--------------------------------------------------------------------------------------
        # 仍保持特征和标签放在 X_features,Y_labels 这两个变量中

        # 查看早产史
        # zaochan=[]
        # for i in range(len(X_features)):
        #     zaochan.append(X_features[i,0,7])
        # print(dict(Counter(zaochan)))

        # 把早产史和高危放到后面
        X_features = np.concatenate(
            (np.array(X_features), np.array(X_features[:, :, 7]).reshape(np.array(X_features[:, :, 7]).shape[0],
                                                                         np.array(X_features[:, :, 7]).shape[1],
                                                                         1)), axis=2)
        print(X_features.shape)

        X_features = np.concatenate(
            (np.array(X_features), np.array(X_features[:, :, 8]).reshape(np.array(X_features[:, :, 8]).shape[0],
                                                                         np.array(X_features[:, :, 8]).shape[1],
                                                                         1)), axis=2)
        print(X_features.shape)

        X_features = np.concatenate(
            (np.array(X_features), np.array(X_features[:, :, 9]).reshape(np.array(X_features[:, :, 9]).shape[0],
                                                                         np.array(X_features[:, :, 9]).shape[1],
                                                                         1)), axis=2)
        print(X_features.shape)

        X_features = np.concatenate(
            (np.array(X_features), np.array(X_features[:, :, 10]).reshape(np.array(X_features[:, :, 10]).shape[0],
                                                                          np.array(X_features[:, :, 10]).shape[1],
                                                                          1)), axis=2)
        print(X_features.shape)

        X_features = np.concatenate(
            (np.array(X_features), np.array(X_features[:, :, 11]).reshape(np.array(X_features[:, :, 11]).shape[0],
                                                                          np.array(X_features[:, :, 11]).shape[1],
                                                                          1)), axis=2)
        print(X_features.shape)

        X_features = np.concatenate(
            (np.array(X_features), np.array(X_features[:, :, 12]).reshape(np.array(X_features[:, :, 12]).shape[0],
                                                                          np.array(X_features[:, :, 12]).shape[1],
                                                                          1)), axis=2)
        print(X_features.shape)

        X_features = np.concatenate(
            (np.array(X_features), np.array(X_features[:, :, 13]).reshape(np.array(X_features[:, :, 13]).shape[0],
                                                                          np.array(X_features[:, :, 13]).shape[1],
                                                                          1)), axis=2)
        print(X_features.shape)

        X_features = np.array(X_features[:, :, 14:])
        print(X_features.shape)

        X_features = np.squeeze(X_features)

        # 标签归一化, 限定范围, 分段缩放[680,2180],[2180,4400],[4400,5050]------------------------------------------------------

        X = list(X_features)
        Y = list(Y_labels)
        x1, x2, x3 = [], [], []
        y1, y2, y3 = [], [], []

        for i in range(len(X)):
            if Y[i] >= 680 and Y[i] <= 2180:  # [680, 2180]
                x1.append(X[i])
                y1.append(Y[i])

            if Y[i] > 2180 and Y[i] <= 4400:  # [2180,4400]
                x2.append(X[i])
                y2.append(Y[i])

            if Y[i] > 4400 and Y[i] <= 5050:  # [4400, 5050]
                x3.append(X[i])
                y3.append(Y[i])

        print('标签限定范围后:')
        weight_category(y1 + y2 + y3)

        fetal_len_interval1 = 2180 - 680  # 定位边界的区间跨度
        fetal_threshold_interval1 = 680

        fetal_len_interval2 = 4400 - 2180
        fetal_threshold_interval2 = 2180

        fetal_len_interval3 = 5050 - 4400
        fetal_threshold_interval3 = 4400

        # 区间密度=不同标签的数量/区间长度
        y1_D = int(500 * len(dict(Counter(y1)).keys()) / (max(y1) - min(y1)))
        y2_D = int(500 * len(dict(Counter(y2)).keys()) / (max(y2) - min(y2)))
        y3_D = int(500 * len(dict(Counter(y3)).keys()) / (max(y3) - min(y3)))

        print('标签的区间密度:', y1_D, y2_D, y3_D)
        print('区间数量:', len(y1), len(y2), len(y3))
        print('标签的区间跨度:', max(y1) - min(y1), max(y2) - min(y2), max(y3) - min(y3))
        print('标签的最小值:', min(y1), min(y2), min(y3))

        # 一次缩放
        y1_new = list(np.squeeze(y1_D * ((np.array(y1) - fetal_threshold_interval1) / fetal_len_interval1)))
        y2_new = list(np.squeeze(y2_D * ((np.array(y2) - fetal_threshold_interval2) / fetal_len_interval2) + y1_D))
        y3_new = list(
            np.squeeze(y3_D * ((np.array(y3) - fetal_threshold_interval3) / fetal_len_interval3) + y1_D + y2_D))

        # 二次缩放前的标签
        pre_rescale = y1_new + y2_new + y3_new
        print('二次缩放前:')
        print('标签的最小值:', min(pre_rescale))
        print('标签的最大值:', max(pre_rescale))
        print('标签的区间跨度:', max(pre_rescale) - min(pre_rescale))

        # 二次缩放
        sum_Temp_fetal = MinMax.fit_transform(np.array(y1_new + y2_new + y3_new).reshape(-1, 1))
        y1_new = list(sum_Temp_fetal[:len(y1_new)])
        y2_new = list(sum_Temp_fetal[len(y1_new):len(y1_new) + len(y2_new)])
        y3_new = list(sum_Temp_fetal[len(y1_new) + len(y2_new):len(y1_new) + len(y2_new) + len(y3_new)])

        X_features = x1 + x2 + x3
        Y_labels = y1_new + y2_new + y3_new

        X_features = np.array(X_features)
        Y_labels = np.array(Y_labels).reshape(-1, 1)

        # print('标签归一化后X:',X_features.shape)
        # print('标签归一化后Y:',Y_labels.shape)

        # 乱序之前转列表
        X_features = list(X_features)
        Y_labels = list(Y_labels)

        # 划分训练集和测试集之前乱序，使得每次抽取的训练集和测试集不相同
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(X_features)
        random.seed(randnum)
        random.shuffle(Y_labels)

        # 乱序之后转回矩阵
        X_features = np.array(X_features)
        Y_labels = np.array(Y_labels)

        print('')
        print('全部数据归一化后特征和标签的维度:')
        print(X_features.shape)
        print(Y_labels.shape)

        # 还原标签, Y_labels_restore中的顺序与Y_labels一致
        Y_labels_restore = []
        for i in range(Y_labels.shape[0]):
            if 86 * Y_labels[i] >= 0 and 86 * Y_labels[i] <= 21:
                Y_labels_restore.append(86 * Y_labels[i] / y1_D * fetal_len_interval1 + fetal_threshold_interval1)

            if 86 * Y_labels[i] > 21 and 86 * Y_labels[i] <= 72:
                Y_labels_restore.append(
                    (86 * Y_labels[i] - y1_D) / y2_D * fetal_len_interval2 + fetal_threshold_interval2)

            if 86 * Y_labels[i] > 72 and 86 * Y_labels[i] <= 86:
                Y_labels_restore.append(
                    (86 * Y_labels[i] - y1_D - y2_D) / y3_D * fetal_len_interval3 + fetal_threshold_interval3)

        # 抽取各类体重胎儿的索引
        low_index = []
        normal_index = []
        high_index = []

        for i in range(len(Y_labels_restore)):
            if Y_labels_restore[i] < 2500:
                low_index.append(i)
            if Y_labels_restore[i] >= 2500 and Y_labels_restore[i] <= 4000:
                normal_index.append(i)
            if Y_labels_restore[i] > 4000:
                high_index.append(i)

        # 将各类体重的索引划分给训练集、测试集
        train_low = low_index[:len(low_index) - 50] * int(len(normal_index) / len(low_index))
        train_normal = normal_index[:len(normal_index) - 50]
        train_high = high_index[:len(high_index) - 50] * int(len(normal_index) / len(high_index))

        test_low = low_index[len(low_index) - 50:]
        test_normal = normal_index[len(normal_index) - 50:]
        test_high = high_index[len(high_index) - 50:]

        # 根据索引将特征和标签放入训练集和测试集的各类列表中
        # 训练集
        X_train_low = []
        Y_train_low = []
        for i in train_low:
            X_train_low.append(X_features[i])
            Y_train_low.append(Y_labels[i])

        X_train_normal = []
        Y_train_normal = []
        for i in train_normal:
            X_train_normal.append(X_features[i])
            Y_train_normal.append(Y_labels[i])

        X_train_high = []
        Y_train_high = []
        for i in train_high:
            X_train_high.append(X_features[i])
            Y_train_high.append(Y_labels[i])

        X_train = np.array(X_train_low + X_train_normal + X_train_high)
        Y_train = np.array(Y_train_low + Y_train_normal + Y_train_high)

        # 划分训练集和测试集之后乱序，使得训练集各类标签顺序不相同
        X_train = list(X_train)
        Y_train = list(Y_train)
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(X_train)
        random.seed(randnum)
        random.shuffle(Y_train)
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        # 测试集
        X_test_low = []
        Y_test_low = []
        for i in test_low:
            X_test_low.append(X_features[i])
            Y_test_low.append(Y_labels[i])

        X_test_normal = []
        Y_test_normal = []
        for i in test_normal:
            X_test_normal.append(X_features[i])
            Y_test_normal.append(Y_labels[i])

        X_test_high = []
        Y_test_high = []
        for i in test_high:
            X_test_high.append(X_features[i])
            Y_test_high.append(Y_labels[i])

        X_test = np.array(X_test_low + X_test_normal + X_test_high)
        Y_test = np.array(Y_test_low + Y_test_normal + Y_test_high)

        X_train = X_train.astype(np.float32)
        Y_train = Y_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        Y_test = Y_test.astype(np.float32)

        print('训练集特征维度:', X_train.shape)
        print('训练集标签维度:', Y_train.shape)
        print('测试集特征维度:', X_test.shape)
        print('测试集标签维度:', Y_test.shape)

        for i in range(len(X_train)):
            count = 0
            for j in range(len(X_train[i])):
                if sum(X_train[i, j, :]) != 0:
                    count += 1
            train_check_num.append(count)

        # 抽取最后一次体检记录作为GBDT的特征
        GBDT_X_train = []
        for i in range(len(X_train)):
            GBDT_X_train.append(X_train[i, train_check_num[i] - 1, :])

        # 测试集孕妇的体检次数
        test_check_num = []
        for i in range(len(X_test)):
            count = 0
            for j in range(len(X_test[i])):
                if sum(X_test[i, j, :]) != 0:
                    count += 1
            test_check_num.append(count)

        # 抽取最后一次体检记录作为SVR的特征
        GBDT_X_test = []
        for i in range(len(X_test)):
            GBDT_X_test.append(X_test[i, test_check_num[i] - 1, :])

        X_train = np.array(GBDT_X_train)
        X_test = np.array(GBDT_X_test)

        gbdt = GradientBoostingRegressor(
            loss='ls',
            learning_rate=0.1,
            n_estimators=100,
            subsample=1,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            alpha=0.9
        )
        # 有更改，加了ravel（）
        gbdt.fit(X_train, Y_train.ravel())
        Y_test_pred = gbdt.predict(X_test)
        print('预测值:', list(Y_test_pred))
        print('真实值:', list(np.squeeze(Y_test)))
        print('')

        # 还原标签
        Y_test = np.array(Y_test) * 86
        Y_test_pred = np.array(Y_test_pred) * 86

        Y_test_pred_restore = []
        for i in range(len(Y_test_pred)):
            if Y_test_pred[i] >= 0 and Y_test_pred[i] <= 21:
                Y_test_pred_restore.append(Y_test_pred[i] / y1_D * fetal_len_interval1 + fetal_threshold_interval1)

            if Y_test_pred[i] > 21 and Y_test_pred[i] <= 72:
                Y_test_pred_restore.append(
                    (Y_test_pred[i] - y1_D) / y2_D * fetal_len_interval2 + fetal_threshold_interval2)

            if Y_test_pred[i] > 72 and Y_test_pred[i] <= 86:
                Y_test_pred_restore.append(
                    (Y_test_pred[i] - y1_D - y2_D) / y3_D * fetal_len_interval3 + fetal_threshold_interval3)

            # 预测越界，分别取样本中的最大值和最小值
            if Y_test_pred[i] < 0:
                Y_test_pred_restore.append(680)
            if Y_test_pred[i] > 86:
                Y_test_pred_restore.append(5165)

        Y_test_real_restore = []
        for i in range(len(Y_test)):
            if Y_test[i] >= 0 and Y_test[i] <= 21:
                Y_test_real_restore.append(Y_test[i] / y1_D * fetal_len_interval1 + fetal_threshold_interval1)

            if Y_test[i] > 21 and Y_test[i] <= 72:
                Y_test_real_restore.append((Y_test[i] - y1_D) / y2_D * fetal_len_interval2 + fetal_threshold_interval2)

            if Y_test[i] > 72 and Y_test[i] <= 86:
                Y_test_real_restore.append(
                    (Y_test[i] - y1_D - y2_D) / y3_D * fetal_len_interval3 + fetal_threshold_interval3)

        Y_test_real_restore = list(np.squeeze(Y_test_real_restore))
        for i in range(len(Y_test_real_restore)):
            Y_test_real_restore[i] = round(Y_test_real_restore[i], 0)
        print('还原后的标签:', list(np.squeeze(Y_test_real_restore)))

        for i in range(len(Y_test_pred_restore)):
            Y_test_pred_restore[i] = round(Y_test_pred_restore[i], 0)
        print('还原后的预测:', Y_test_pred_restore)

        total_MRE = []
        total_MAE = []
        total_ACC = []
        for i in range(len(Y_test_real_restore)):
            total_MRE.append(abs(Y_test_real_restore[i] - Y_test_pred_restore[i]) / Y_test_real_restore[i])
            total_MAE.append(abs(Y_test_real_restore[i] - Y_test_pred_restore[i]))
            if abs(Y_test_real_restore[i] - Y_test_pred_restore[i]) <= 250:
                total_ACC.append(1)
            else:
                total_ACC.append(0)

        print('总体平均相对误差:', sum(total_MRE) / len(total_MRE))
        print('总体平均绝对误差:', sum(total_MAE) / len(total_MAE))
        print('总体准确率:', sum(total_ACC) / len(total_ACC))
        res_list.append('总体平均相对误差:'+ str(sum(total_MRE) / len(total_MRE)))
        res_list.append('总体平均绝对误差:'+ str(sum(total_MAE) / len(total_MAE)))
        res_list.append('总体准确率:'+ str(sum(total_ACC) / len(total_ACC)))
        low_MRE = []
        low_MAE = []
        low_ACC = []
        for i in range(0, 50):  # 低体重儿的索引
            low_MRE.append(abs(Y_test_real_restore[i] - Y_test_pred_restore[i]) / Y_test_real_restore[i])
            low_MAE.append(abs(Y_test_real_restore[i] - Y_test_pred_restore[i]))
            if abs(Y_test_real_restore[i] - Y_test_pred_restore[i]) <= 250:
                low_ACC.append(1)
            else:
                low_ACC.append(0)
        print('低体重平均相对误差:', sum(low_MRE) / len(low_MRE))
        print('低体重平均绝对误差:', sum(low_MAE) / len(low_MAE))
        print('低体重准确率:', sum(low_ACC) / len(low_ACC))
        res_list.append('低体重平均相对误差:' + str(sum(low_MRE) / len(low_MRE)))
        res_list.append('低体重平均绝对误差:' + str(sum(low_MAE) / len(low_MAE)))
        res_list.append('低体重准确率:' + str(sum(low_ACC) / len(low_ACC)))
        normal_MRE = []
        normal_MAE = []
        normal_ACC = []
        for i in range(50, 100):
            normal_MRE.append(abs(Y_test_real_restore[i] - Y_test_pred_restore[i]) / Y_test_real_restore[i])
            normal_MAE.append(abs(Y_test_real_restore[i] - Y_test_pred_restore[i]))
            if abs(Y_test_real_restore[i] - Y_test_pred_restore[i]) <= 250:
                normal_ACC.append(1)
            else:
                normal_ACC.append(0)
        print('正常体重平均相对误差:', sum(normal_MRE) / len(normal_MRE))
        print('正常体重平均绝对误差:', sum(normal_MAE) / len(normal_MAE))
        print('正常体重准确率:', sum(normal_ACC) / len(normal_ACC))
        res_list.append('正常体重平均相对误差:' + str(sum(normal_MRE) / len(normal_MRE)))
        res_list.append('正常体重平均绝对误差:' + str(sum(normal_MAE) / len(normal_MAE)))
        res_list.append('正常体重准确率:' + str(sum(normal_ACC) / len(normal_ACC)))
        high_MRE = []
        high_MAE = []
        high_ACC = []
        for i in range(100, 150):
            high_MRE.append(abs(Y_test_real_restore[i] - Y_test_pred_restore[i]) / Y_test_real_restore[i])
            high_MAE.append(abs(Y_test_real_restore[i] - Y_test_pred_restore[i]))
            if abs(Y_test_real_restore[i] - Y_test_pred_restore[i]) <= 250:
                high_ACC.append(1)
            else:
                high_ACC.append(0)
        print('高体重平均相对误差:', sum(high_MRE) / len(high_MRE))
        print('高体重平均绝对误差:', sum(high_MAE) / len(high_MAE))
        print('高体重准确率:', sum(high_ACC) / len(high_ACC))
        res_list.append('高体重平均相对误差:' + str(sum(high_MRE) / len(high_MRE)))
        res_list.append('高体重平均绝对误差:' + str(sum(high_MAE) / len(high_MAE)))
        res_list.append('高体重准确率:' + str(sum(high_ACC) / len(high_ACC)))
        # 画混淆矩阵
        low2low, low2normal, low2high = 0, 0, 0
        normal2low, normal2normal, normal2high = 0, 0, 0
        high2low, high2normal, high2high = 0, 0, 0

        for i in range(0, 50):
            if Y_test_pred_restore[i] < 2500:
                low2low += 1
            if Y_test_pred_restore[i] >= 2500 and Y_test_pred_restore[i] <= 4000:
                low2normal += 1
            if Y_test_pred_restore[i] > 4000:
                low2high += 1

        for i in range(50, 100):
            if Y_test_pred_restore[i] < 2500:
                normal2low += 1
            if Y_test_pred_restore[i] >= 2500 and Y_test_pred_restore[i] <= 4000:
                normal2normal += 1
            if Y_test_pred_restore[i] > 4000:
                normal2high += 1

        for i in range(100, 150):
            if Y_test_pred_restore[i] < 2500:
                high2low += 1
            if Y_test_pred_restore[i] >= 2500 and Y_test_pred_restore[i] <= 4000:
                high2normal += 1
            if Y_test_pred_restore[i] > 4000:
                high2high += 1

        confusion_matrix = [[low2low, low2normal, low2high],
                            [normal2low, normal2normal, normal2high],
                            [high2low, high2normal, high2high]]

        fig, ax = plt.subplots(figsize=(9, 9))
        cmap = mcolors.LinearSegmentedColormap.from_list('n', ['#ffff99', '#ff0099'])
        sns.heatmap(pd.DataFrame(confusion_matrix,
                                 columns=['低体重儿', '正常体重儿','巨大儿'],
                                 index=['低体重儿', '正常体重儿','巨大儿']),annot=True, fmt='d',vmax=50, vmin=0, cmap=cmap,
                    square=True, linewidths=0.01, linecolor='white', cbar=False)

        plt.gca().xaxis.set_ticks_position('top')
        plt.xticks(size=12)
        plt.yticks(size=12, rotation=0)
        plt.savefig('./static/images/demo.png')
        return render(request,"run.html",{'res_list':res_list})


# 计算每个数据点到其聚类中心的距离
def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0, len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]]
        distance.set_value(i, np.linalg.norm(Xa - Xb))
    return distance
 plt=None
def kmeans(self,n):
    # 设置异常值比例
    outliers_fraction = 0.01
    # 得到每个点到取聚类中心的距离，我们设置了N个聚类中心
    distance = getDistanceByPoint(data, kmeans[n])
    # 根据异常值比例outliers_fraction计算异常值的数量
    number_of_outliers = int(outliers_fraction * len(distance))
    # 设定异常值的阈值
    threshold = distance.nlargest(number_of_outliers).min()
    # 根据阈值来判断是否为异常值
    df['anomaly1'] = (distance >= threshold).astype(int)
    # 数据可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df["anomaly1"].apply(lambda x: colors[x]))
    plt.xlabel('num')
    plt.ylabel('num')
    plt.show()