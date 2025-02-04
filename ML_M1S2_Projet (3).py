#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

file_path = "C:/Users/lenovo/Desktop/九大M1S1/机器学习--M1S1/bank+marketing/bank-full.csv"
data=pd.read_csv(file_path, sep=';', quotechar='"')
print(data.head())
print(data.info())

data.to_csv('C:/Users/lenovo/Desktop/bank-full_test.csv')


# In[68]:


#开始进行数据清理

data = data.drop_duplicates()
print("删除重复值后的数据大小:", data.shape)


# In[69]:


data.isnull().sum()


# In[70]:


data.describe()


# In[71]:


data.describe(include='O')


# In[72]:


#绘制箱型图，寻找是否有异常值
import matplotlib.pyplot as plt

numerical_columns = data.select_dtypes(include = ['int64','float64']).columns

plt.figure(figsize = (10,10))

for i, j in enumerate(numerical_columns, 1):
        plt.subplot((len(numerical_columns) + 2) // 3, 3, i)
        plt.boxplot(data[j], vert = False)
        plt.title(j)
        
plt.tight_layout()
plt.show()
    


# 发现在balance，previous，duratin和campaign中发现一些极高的异常值

# In[73]:


#检查异常值与目标变量y的关系，若不显著则删除

#数值型
import seaborn as sns

plt.figure(figsize=(10, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot((len(numerical_columns) + 2) // 3, 3, i)
    sns.boxplot(x='y', y=col, data=data)
    plt.title(f'{col} vs y')

plt.tight_layout()
plt.show()


# 经过对比，发现previous中有个很离谱的值，但是对于整体分布无明显影响，先删了
# 
# 发现，在DAY特征下，yes与no区别不明显，可以先删除，不作为一个特征。
# 

# In[77]:


#删除previous中>100，以及day

data = data[data['previous'] <= 100]
data = data.drop(columns = ['day'])


print(f"删除后: {data.shape}")


# In[78]:


sns.boxplot(x = 'y', y = 'previous', data = data)
plt.title('previous vs y')
plt.show()


# In[93]:


#进行下一步特征工程
#首先进行非数值类特征与y的关系图

#重置索引，因为删除了day
data.reset_index(drop = True, inplace = True)

print(data.head())

#绘制非数值类关系图
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 
                       'loan', 'contact', 'month', 'poutcome']
f = pd.melt(data, id_vars = 'y', value_vars = categorical_columns)

# 使用 FacetGrid 绘制柱状图
g = sns.FacetGrid(f, col = 'variable', col_wrap = 3, height = 5, sharex = False, sharey = False, hue = 'y')
g.map(sns.histplot, 'value', stat = 'density', multiple = 'dodge', shrink = 0.8)

g.add_legend()
plt.savefig('分类特征分布柱状图.png', bbox_inches = 'tight')
plt.show()


# 每幅图中横轴代表不同特征值，颜色代表y的值。不同柱子中的相同颜色相加等于1。
# 浅橙色代表yes，深橙色部分是蓝色和浅橙色重叠了
# 
# 对比发现，defaut特征中，不同特征值对目标变量y几乎没有分辨能力，所以删除。

# In[94]:


data = data.drop(columns = ['default'])


# In[96]:


# 现在将非数值类型的特征值变为数值类型
categorical_columns = ['job', 'marital', 'education', 'housing', 'loan', 
                       'contact', 'month', 'poutcome', 'y']

# 创建映射字典并应用映射
mappings = {}
for col in categorical_columns:
    # 为每个列创建字典
    unique_values = data[col].unique()
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    reverse_mapping = {idx: value for value, idx in mapping.items()}
    
    # 存储映射和反转字典
    mappings[col] = {'mapping': mapping, 'reverse_mapping': reverse_mapping}
    
    # 应用映射，将类别转换为数值
    data[col] = data[col].map(mapping)

print(data.head())


# In[97]:


# 检查
print("Job 映射:", mappings['job']['mapping'])
print("Job 值的唯一值:", data['job'].unique())


# In[99]:


#现在完成了所有数值化，接下来进行标准化操作
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

features = data.drop(columns = ['y'])
features_scaled = scaler.fit_transform(features)

data_scaled = pd.DataFrame(features_scaled, columns = features.columns)
data_scaled['y'] = data['y']
print(data_scaled.head())


# In[102]:


#开始进一步选择特征

#使用随机森林评估不同特征的重要性
from sklearn.ensemble import RandomForestClassifier

x = data_scaled.drop(columns = ['y'])
y = data_scaled['y']

model = RandomForestClassifier(random_state = 42)
model.fit(x,y)

#显示出特征值的重要性
importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': importance})
importance_df = importance_df.sort_values(by = 'Importance', ascending = False)
print( importance_df )

#可视化
plt.figure(figsize = (10,10))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title('Feature Importance')
plt.show()



# In[105]:


#由于特征太多，所以先选择贡献最大的10个
selected_features = importance_df.head(10)['Feature'].tolist()
print("选择的Features: ", selected_features)

x_selected = data_scaled[selected_features]


# In[104]:


#由于特征太多，所以先选择贡献最大的10个
from sklearn.decomposition import PCA

pca = PCA(n_components = 0.95)
x_pca = pca.fit_transform(x)

print("降维后还剩:", x_pca.shape[1])


# In[ ]:




