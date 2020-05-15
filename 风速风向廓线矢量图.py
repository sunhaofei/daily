import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel('/Users/sunhaofei/Downloads/10.16-10.21.xlsx', sheet_name ='16')
data2 = pd.read_excel('/Users/sunhaofei/Downloads/10.16-10.21.xlsx', sheet_name ='17')
data3 = pd.read_excel('/Users/sunhaofei/Downloads/10.16-10.21.xlsx', sheet_name ='18')
data4 = pd.read_excel('/Users/sunhaofei/Downloads/10.16-10.21.xlsx', sheet_name ='19')
data5 = pd.read_excel('/Users/sunhaofei/Downloads/10.16-10.21.xlsx', sheet_name ='20')
data6 = pd.read_excel('/Users/sunhaofei/Downloads/10.16-10.21.xlsx', sheet_name ='21')
# print(data)

def u_v(wd, ws):
        u = -ws * np.sin(wd*np.pi/180)
        v = -ws * np.cos(wd*np.pi/180)
        return u, v

data["u1"], data["v1"] = u_v(data["wd1"], data["ws1"])
data["u2"], data["v2"] = u_v(data["wd2"], data["ws2"])

data2["u1"], data2["v1"] = u_v(data2["wd1"], data2["ws1"])
data2["u2"], data2["v2"] = u_v(data2["wd2"], data2["ws2"])

data3["u1"], data3["v1"] = u_v(data3["wd1"], data3["ws1"])
data3["u2"], data3["v2"] = u_v(data3["wd2"], data3["ws2"])

data4["u1"], data4["v1"] = u_v(data4["wd1"], data4["ws1"])
data4["u2"], data4["v2"] = u_v(data4["wd2"], data4["ws2"])

data5["u1"], data5["v1"] = u_v(data5["wd1"], data5["ws1"])
data5["u2"], data5["v2"] = u_v(data5["wd2"], data5["ws2"])

data6["u1"], data6["v1"] = u_v(data6["wd1"], data6["ws1"])
data6["u2"], data6["v2"] = u_v(data6["wd2"], data6["ws2"])

# print(data)
# print(data2)
fig = plt.figure(figsize=(13,15))

axs1 = fig.add_subplot(231)
axs1.quiver(data['x2'], data['z2'], data['u2'], data['v2'],scale=80,headwidth=5 )
axs1.quiver(data['x1'], data['z1'], data['u1'], data['v1'],scale=80,headwidth=5 )
axs1.set_xlim(0,6)
axs1.set_ylim(0,2000)
plt.ylabel('Height')
xticks = ['04','08','12','16','20','24']
plt.xticks([1,2,3,4,5,6],xticks)

axs2 = fig.add_subplot(232)
axs2.quiver(data2['x2'], data2['z2'], data2['u2'], data2['v2'],scale=80,headwidth=5 )
axs2.quiver(data2['x1'], data2['z1'], data2['u1'], data2['v1'],scale=80,headwidth=5 )
axs2.set_xlim(0,6)
axs2.set_ylim(0,2000)
# plt.ylabel('Height')
xticks = ['04','08','12','16','20','24']
plt.xticks([1,2,3,4,5,6],xticks)

axs3 = fig.add_subplot(233)
axs3.quiver(data3['x2'], data3['z2'], data3['u2'], data3['v2'],scale=80,headwidth=5 )
axs3.quiver(data3['x1'], data3['z1'], data3['u1'], data3['v1'],scale=80,headwidth=5 )
axs3.set_xlim(0,6)
axs3.set_ylim(0,2000)
xticks = ['04','08','12','16','20','24']
plt.xticks([1,2,3,4,5,6],xticks)

axs4 = fig.add_subplot(234)
axs4.quiver(data4['x2'], data4['z2'], data4['u2'], data4['v2'],scale=80,headwidth=5 )
axs4.quiver(data4['x1'], data4['z1'], data4['u1'], data4['v1'],scale=80,headwidth=5 )
axs4.set_xlim(0,6)
axs4.set_ylim(0,2000)
plt.ylabel('Height')
xticks = ['04','08','12','16','20','24']
plt.xticks([1,2,3,4,5,6],xticks)

axs5 = fig.add_subplot(235)
axs5.quiver(data5['x2'], data5['z2'], data5['u2'], data5['v2'],scale=80,headwidth=5 )
axs5.quiver(data5['x1'], data5['z1'], data5['u1'], data5['v1'],scale=80,headwidth=5 )
axs5.set_xlim(0,6)
axs5.set_ylim(0,2000)
# plt.ylabel('Height')
xticks = ['04','08','12','16','20','24']
plt.xticks([1,2,3,4,5,6],xticks)

axs6 = fig.add_subplot(236)
axs6.quiver(data6['x2'], data6['z2'], data6['u2'], data6['v2'],scale=80,headwidth=5 )
axs6.quiver(data6['x1'], data6['z1'], data6['u1'], data6['v1'],scale=80,headwidth=5 )
axs6.set_xlim(0,6)
axs6.set_ylim(0,2000)
xticks = ['04','08','12','16','20','24']
plt.xticks([1,2,3,4,5,6],xticks)


plt.savefig('/Users/sunhaofei/Downloads/1.pdf')
plt.show()