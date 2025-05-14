import data_preprocessing
import MLP
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 数据预处理
    data = data_preprocessing.data_preprocessing(r'dataset\user_data.csv')
    # 测试集与训练集
    X_train, X_test, y_train, y_test = data_preprocessing.split_data(data)
    X_train = X_train.values.T
    y_train = y_train.values
    X_test = X_test.values.T
    y_test = y_test.values
    # 训练模型
    input_size = X_train.shape[0]  # 输入特征的维度
    output_size = 4 # 输出特征的维度

    # 超参数
    epochs = [10, 100]
    alphas = [0.5, 0.1]
    for i in epochs:
        for j in alphas:
            print("学习率 = %.2f，迭代次数 = %d" % (j, i))

            # 训练模型
            model = MLP.Nerual_Network(input_size, 12, output_size, j, i)
            model.train(X_train, y_train)

            # 预测
            model.predict(X_test, y_test)

            # 可视化损失函数
            model.visualize_loss()