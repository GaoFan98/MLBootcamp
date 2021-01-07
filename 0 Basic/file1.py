import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# my_list = [[1,2,3],[4,5,6],[7,8,9]]
#
# print(np.array(my_list))

# print(np.arange(0,11,2))

# print(np.zeros(5))
# print(np.zeros((5,6)))

# print(np.linspace(0,2,10,False))

# print(np.eye(4))

# print(np.random.rand(5,5))
# print(np.random.randn(5,5))
# print(np.random.randint(1,5,5))

# arr = np.arange(20)
# print(arr.reshape(5,4))
# print(arr.shape)

# arr = np.arange(20)
# print(arr[arr != 5])

# arr = [2,4,-2]
# print(np.log10(10))

# labels = ['a','b','c']
# data = {'a':10, 'b':20, 'c':30}
#
# print(pd.Series(data))

# from numpy.random import randn

# np.random.seed(101)

# df = pd.DataFrame(randn(5,4),['a','b','c','d','e'],['w','x','y','z'])
# df['new'] = [1,2,4,5,3]
# df.drop('new',1,inplace=False)
# print(df['new'])
# df.drop('b',axis=0,inplace=True)
# print(df)
# print(df.loc['c'])
# print(df.iloc[2])

# print(df.loc['d','z'])

# print(df)
# print(df[df['w']<0])

# print(df)
# print((df['w']<0) | (df['w']<0))

# a = df.set_index('w')
# print(a)

# outside = ['G1', 'G1', 'G2', 'G2', 'G3', 'G3']
# inside = [1, 2, 3, 4, 1, 6]
# hier_index = list(zip(outside, inside))
# hier_index = pd.MultiIndex.from_tuples(hier_index)
# # print(hier_index)
#
# df = pd.DataFrame(randn(6, 2), hier_index, ['A', 'B'])
# # print(df.loc['G3','A'].loc[5])
#
# df.index.names = ['Groups', 'Nums']
# print(df)
#
# print(df.xs(1, level='Nums'))

# d = {'A':[1,2,np.nan],'B':[4,np.nan,np.nan],'C':['hello',2,4]}
#
# df = pd.DataFrame(d)
# print(df.dropna(0))
# print(df.dropna(thresh=2))
# print(df.fillna(value='FIll'))
# print(df['A'].fillna(value=df['A'].mean()))
#
# data =  {
#     'Company' : ['Google','Microsoft','Google','Facebook','Microsoft','Google'],
#     'Person' : ['Bob','Chester','Susan','Katy','Lin','Lary'],
#     'Salary' : [200,100,500,600,340,890]
# }

# df = pd.DataFrame(data)
# group_by_comp = df.groupby('Company')
# print(group_by_comp.mean())
# print(group_by_comp.sum().loc['Google'])
# print(df.groupby('Company').sum().loc['Google'])
# print(df.groupby('Company').count().loc['Google'])
# print(df.groupby('Company').max().loc['Google'])
# print(df)
# print(df.groupby('Company').describe().loc['Google'])
# print(df.groupby('Company').describe().transpose())

# x = np.linspace(0, 5, 11)
# y = x ** 2

# print(plt.plot(x,y,'r'))

# fig = plt.figure()
# axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# axes.plot(x, y,'r')

# fig, axes = plt.subplots(2,2)

# plt.show()
