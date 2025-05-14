import pandas as pd
from sklearn.model_selection import train_test_split

'''
user_data包含如下字段：
 字段           字段说明            
 user_id       用户ID              
#item_id       商品ID              
 behavior_type 用户对商品的行为类型   包括浏览、收藏、加入购物车、购买，对应取值分别是1、2、3、4（作为Y）
 item_category 商品分类ID           
 time          行为时间             精确到小时级别

item_data包含如下字段：
 字段           字段说明          
 item_id       商品ID             
 item_category 商品分类ID 
'''
def data_preprocessing(path):
    """
    数据预处理函数
    :param path: 数据文件路径
    :return: 处理后的数据集
    """
    # 设置显示选项，显示最大列数不受限
    pd.set_option('display.max_columns', None)

    # 读取数据
    data = pd.read_csv(path)
    ifdata = pd.read_csv(r'dataset\item_data.csv')
    print(data.head(5))
    print('数据集未处理的大小为：', data.shape)

    # 数据清洗
    '''
    对于user_id，缺失值用上一个数据的user_id填充
    对于item_id，缺失值用上一个数据的item_id填充
    对于behavior_type，缺失值用1（即浏览）填充
    对于item_category，缺失值用该数据的用item_id对应item_data中的item_id所对应的item_category填充，若item_id在item_data中不存在，则用-1填充
    对于time，处理时间，分级为年、月、日、时，缺失值用同一个user_id下的平均值填充
    '''
    ## 缺失值处理
    ### 对于user_id，缺失值用上一个数据的user_id填充
    data['user_id'] = data['user_id'].ffill()
    ### 对于item_id，缺失值用上一个数据的item_id填充
    data['item_id'] = data['item_id'].ffill()
    ### 对于behavior_type，缺失值用1（即浏览）填充
    data['behavior_type'] = data['behavior_type'].fillna(1)
    ### 对于item_category，缺失值用该数据的用item_id对应item_data中的item_id所对应的item_category填充，若item_id在item_data中不存在，则用-1填充
    data['item_category'] = data['item_category'].fillna(-1)
    data['item_category'] = data['item_category'].fillna(data['item_id'].map(ifdata.set_index('item_id')['item_category']))
    ### 处理时间，分级为年、月、日、时
    data['time'] = pd.to_datetime(data['time'])
    data['year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data['day'] = data['time'].dt.day
    data['hour'] = data['time'].dt.hour
    data.drop(columns=['time'], inplace=True)
    ### 对于time，缺失值用同一个user_id下的平均值填充
    data['year'] = data.groupby('user_id')['year'].transform(lambda x: x.fillna(x.mean()))
    data['month'] = data.groupby('user_id')['month'].transform(lambda x: x.fillna(x.mean()))
    data['day'] = data.groupby('user_id')['day'].transform(lambda x: x.fillna(x.mean()))
    data['hour'] = data.groupby('user_id')['hour'].transform(lambda x: x.fillna(x.mean()))
    # print(data.head(5))
    ## 删除重复数据
    data.drop_duplicates(inplace=True)

    # 数据规约
    '''
    对于商品ID，无需计算，可通过商品种类进行预测用户的喜好，推送该种类的商品
    删除商品ID项
    '''
    data.drop(columns=['item_id'], inplace=True)

    # 数据转换
    '''
    用户id为无序离散型数据，用频率编码
    商品分类为无序离散型数据，用哈希编码
    行为类型为离散型数据，作为分类结果，4种分类结果
    时间为离散属性，进行Z-Score标准化，但年用最大最小标准化
    '''
    ## 用户id为无序离散型数据，用频率编码
    user_freq = data['user_id'].value_counts().to_dict()
    data['user_freq'] = data['user_id'].map(user_freq)
    data.drop(columns=['user_id'], inplace=True)
    ## 商品分类为无序离散型数据，用频率编码
    item_freq = data['item_category'].value_counts().to_dict()
    data['item_freq'] = data['item_category'].map(item_freq)
    data.drop(columns=['item_category'], inplace=True)
    ## 时间属性，进行Z-Score标准化，但年用最大最小标准化
    if data['year'].max() - data['year'].min() == 0:
        # 说明年对数据集没有影响
        data.drop(columns=['year'], inplace=True)
    data['month'] = (data['month'] - data['month'].mean()) / data['month'].std()
    data['day'] = (data['day'] - data['day'].mean()) / data['day'].std()
    data['hour'] = (data['hour'] - data['hour'].mean()) / data['hour'].std()

    # 保证类别从0 ~ n - 1
    data['behavior_type'] = data['behavior_type'] - data['behavior_type'].min()

    print(data.head(5))
    print('数据集预处理后的大小为：', data.shape)
    return data

# 拆分数据集，测试集与训练集
def split_data(data):
    """
    拆分数据集，测试集与训练集
    :param data: 数据集
    :return: 训练集和测试集
    """
    y = data['behavior_type']
    X = data.drop(columns=['behavior_type'])
    # 拆分数据集训练集与测试集的比例为7 : 3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    print('这是数据预处理模块')