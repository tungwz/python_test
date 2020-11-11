from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

a = Dataset('Us-Blo-NET-clm.nc')  # 读取nc文件
b = Dataset('Us-Blo-NET-colm.nc')  # 读取nc文件

albd1 = a.variables['f_albd']  # 变量：反射率
albd2 = b.variables['f_albd']  # 变量：反射率
albd3 = a.variables['f_albi']
albd4 = b.variables['f_albi']

alb_vis = []
alb1 = []
alb2 = []
alb3 = []
alb4 = []
alb_nir = []

ali_vis = []
ali1 = []
ali2 = []
ali3 = []
ali4 = []
ali_nir = []

for i in range(60):  # 根据nc文件修改范围

    alb1.append(albd1[i][0][0][0])  # 变量：[记录][波段][经度][纬度]
    alb2.append(albd2[i][0][0][0])  # 变量：[记录][波段][经度][纬度]
    alb_vis.append(albd2[i][0][0][0] - albd1[i][0]  # 两个变量的差
                   [0][0])
    alb3.append(albd1[i][1][0][0])  # 变量：[记录][波段][经度][纬度]
    alb4.append(albd2[i][1][0][0])  # 变量：[记录][波段][经度][纬度]
    alb_nir.append(albd2[i][1][0][0] - albd1[i][1]  # 两个变量的差
                   [0][0])

    ali1.append(albd3[i][0][0][0])
    ali2.append(albd4[i][0][0][0])
#    ali_vis.append(ali2[j][0][0][0] - ali1[j][0][0][0])
    ali3.append(albd3[i][1][0][0])
    ali4.append(albd4[i][1][0][0])
#    ali_nir.append(ali4[j][1][0][0] - ali3[j][1][0][0])

#j = 0
# for i in alb1:
#    if i < 0.:
#        print(i)

#print(max(alb), j)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(2, 2, 1)
ax.plot(alb1, c='b', label='CLM')
ax.plot(alb2, '--', c='r', label='CoLM')
plt.ylim(min(min(alb1), min(alb2)), max(max(alb1), max(alb2))+0.005)
plt.xticks(np.linspace(0, 60, 6))
ax.set_xticklabels(['2001', '2002', '2003', '2004', '2005', '2006'],
                   rotation=45, fontsize='small')
plt.title('a')
plt.ylabel('averaged albedo direct(%)')
plt.legend()
#plt.legend(loc='upper left')
#ax2 = fig.add

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(alb3, c='b', label='CLM')
ax2.plot(alb4, '--', c='r', label='CoLM')
plt.ylim(min(min(alb3), min(alb4)), max(max(alb3), max(alb4)+0.03))
plt.xticks(np.linspace(0, 60, 6))
ax2.set_xticklabels(['2001', '2002', '2003', '2004', '2005', '2006'],
                    rotation=45, fontsize='small')
plt.title('b')
plt.ylabel('averaged albedo direct(%)')
plt.legend()


ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(ali1, c='b', label='CLM')
ax3.plot(ali2, '--', c='r', label='CoLM')
plt.ylim(min(min(ali1), min(ali2)), max(max(ali1), max(ali2))+0.005)
plt.xticks(np.linspace(0, 60, 6))
ax3.set_xticklabels(['2001', '2002', '2003', '2004', '2005', '2006'],
                    rotation=45, fontsize='small')
plt.title('c')
plt.ylabel('averaged albedo diffuse(%)')
plt.legend()

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(ali3, c='b', label='CLM')
ax4.plot(ali4, '--', c='r', label='CoLM')
plt.ylim(min(min(ali3), min(ali4)), max(max(ali3), max(ali4)+0.04))
plt.xticks(np.linspace(0, 60, 6))
ax4.set_xticklabels(['2001', '2002', '2003', '2004', '2005', '2006'],
                    rotation=45, fontsize='small')
plt.title('d')
plt.ylabel('averaged albedo diffuse(%)')
plt.legend()

plt.show()
