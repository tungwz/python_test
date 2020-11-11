import pandas as pd
import os
import glob

fdir = os.listdir('C:/Users/adminX/Desktop/2016/')  # 读取数据路径
outputfile = 'C:/Users/adminX/Desktop/2016/all.csv'  # 输出目录
os.chdir('C:/Users/adminX/Desktop/2016/')  # 修改成数据目录

f = open(fdir[0])
df = pd.read_csv(f)
fnum = glob.glob(pathname='5*.csv')  # pathname根据数据修改
df.to_csv(outputfile, encoding="gb2312", index=False)

for i in range(1, len(fnum)):
    f = open(fdir[i])
    df = pd.read_csv(f)
    df.to_csv(outputfile, encoding="gb2312",
              index=False, header=False, mode='a+')

f = open('all.csv')
df = pd.read_csv(f, header=0)
df = df.drop_duplicates(keep='first')
df = df.to_csv('new.csv', encoding='gb2312', index=False, header=True)
