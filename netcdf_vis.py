from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import matplotlib.pyplot as plt

a = Dataset('old_op_2D_Fluxes_2012-12.nc')  # 读取nc文件
b = Dataset('old_op_2D_Fluxes_2003-01.nc')  # 读取nc文件
c = Dataset('old_op_2D_Fluxes_2003-02.nc')
a1 = Dataset('new_op_2D_Fluxes_2012-12.nc')  # 读取nc文件
b1 = Dataset('new_op_2D_Fluxes_2003-01.nc')  # 读取nc文件
c1 = Dataset('new_op_2D_Fluxes_2003-02.nc')

albd1 = a.variables['f_albd']  # 变量：albedo
albd2 = b.variables['f_albd']  # 变量：albedo
albd3 = c.variables['f_albd']  # 变量：albedo
albd4 = a1.variables['f_albd']  # 变量：albedo
albd5 = b1.variables['f_albd']  # 变量：albedo
albd6 = c1.variables['f_albd']  # 变量：albedo

alb_vis = []
alb1 = []
alb2 = []
alb3 = []
alb4 = []
alb_nir = []
for i in range(60):  # 根据nc文件修改范围

    alb1.append(albd1[i][0][0][0])  # 变量：[记录][波段][经度][纬度]
    alb2.append(albd2[i][0][0][0])  # 变量：[记录][波段][经度][纬度]
    alb_vis.append(albd2[i][0][0][0] - albd1[i][0]  # 两个变量的差
                   [0][0])

for i in range(60):  # 根据nc文件修改范围

    alb3.append(albd1[i][1][0][0])  # 变量：[记录][波段][经度][纬度]
    alb4.append(albd2[i][1][0][0])  # 变量：[记录][波段][经度][纬度]
    alb_nir.append(albd2[i][1][0][0] - albd1[i][1]  # 两个变量的差
                   [0][0])
#j = 0
# for i in alb1:
#    if i < 0.:
#        print(i)

#print(max(alb), j)
plt.figure(figsize=(10, 5))
plt.subplot(411)
plt.plot(alb1, c='b', label='CLM')
plt.plot(alb2, '--', c='r', label='CoLM')
plt.figure.add_subplot.set_xticklabels(['2001', '2002', '2003', '2004', '2005'],
                                       rotation=45, fontsize='small')
plt.suptitle('f_albd(visiable)')
plt.ylabel('averaged albedo direct(%)')
plt.xlabel('time')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(alb_vis)
plt.xlabel('time')

plt.subplot(413)
plt.plot(alb3, c='b', label='CLM')
plt.plot(alb4, '--', c='r', label='CoLM')
plt.suptitle('f_albd(nir)')
plt.ylabel('averaged albedo direct(%)')
plt.xlabel('time')
plt.legend(loc='upper left')
plt.savefig('albd_nir_Var.png')

plt.subplot(414)
plt.plot(alb_nir)
plt.xlabel('time')

plt.show()
