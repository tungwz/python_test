# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:33:16 2020

@author: ZongrAx
"""

import random
import xlrd

f = xlrd.open_workbook('group.xlsx')  # 分组名单

sheet = f.sheets()[0]
cols1 = sheet.col_values(0)
cols2 = sheet.col_values(1)

NN = 3  # 每组最大组队人数
N = int((len(cols1))/NN)
# print(N)

m = [[0 for i in range(NN)] for i in range(N)]

for i in range(N):   #
    for j in range(NN):
        mm = i+i*2+j
        m[i][j] = cols1[mm]
# print(m)

x = list(range(N))
xx = list(range(NN))
cc = 12  # 队长人数

cnum = random.sample(x, cc)
cap = list(range(12))  # 组长
print(cnum)

i = 0
for n in cnum:
    #n = random.choice(x)
    nn = random.choice(xx)
    # print(n)
    # print(nn)
    while m[n][nn] == 0:
        #n = random.choice(x)
        nn = random.choice(xx)

    cap[i] = m[n][nn]
    i += 1

    del m[n][:]
    x.remove(n)
# print(m)
# print(x)
# print(cap)

for i in cap:
    ff = cols1.index(int(i))
    aa = sheet.row_values(ff)
    print(str(int(aa[0]))+' '+aa[1])
    #print(str(int(aa[0]))+' '+aa[1].encode('utf-8'))
