# find the maximum and minimum of the discrete data set
# 寻找离散序列极值点的方法

# https://blog.csdn.net/weijifen000/article/details/80070520?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-2
import numpy as np 
import pylab as pl
import matplotlib.pyplot as plt
import scipy.signal as signal
x=np.array([
    0, 6, 25, 20, 15, 8, 15, 6, 0, 6, 0, -5, -15, -3, 4, 10, 8, 13, 8, 10, 3,
    1, 20, 7, 3, 0 ])
plt.figure(figsize=(16,4))
plt.plot(np.arange(len(x)),x)
print x[signal.argrelextrema(x, np.greater)]
print signal.argrelextrema(x, np.greater)

plt.plot(signal.argrelextrema(x,np.greater)[0],x[signal.argrelextrema(x, np.greater)],'o')
plt.plot(signal.argrelextrema(-x,np.greater)[0],x[signal.argrelextrema(-x, np.greater)],'+')
# plt.plot(peakutils.index(-x),x[peakutils.index(-x)],'*')
plt.show()

###########################################################
# when it comes to "dataframe"
import numpy as np
import pandas as pd
import scipy.signal as signal

df = pd.DataFrame(columns=['grad_theta_smooth'])
df['grad_theta_smooth'] = np.array([
    0, 6, 25, 20, 15, 8, 15, 6, 0, 6, 0, -5, -15, -3, 4, 10, 8, 13, 8, 10, 3,
    1, 20, 7, 3, 0])
l = np.array(df['grad_theta_smooth'])[signal.argrelextrema(np.array(df['grad_theta_smooth']), np.greater)][0] # 极大值
s = np.array(df['grad_theta_smooth'])[signal.argrelextrema(np.array(df['grad_theta_smooth']), np.less_equal)][1] # 极小值
# greater与greater的差别在于：greater在极值有左右相同点的时候无法识别；而greater_equal可以。
print(l)
print(s)
