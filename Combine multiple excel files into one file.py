
## Combine multiple Excel files and multiple sheets into one Excel file
## From the URL:https://blog.csdn.net/d1240673769/article/details/74513206
## But it needs to specify the file name manually, 
## I made some changes so that I can operate all the Excel files in the folder
## Unfortunately, the files in the folder must all be in ".xlsx" format
## Note，the merged file is in reverse order of the files in the folder
# -*- coding: utf-8 -*-

#将多个Excel文件合并成一个
import xlrd
import xlsxwriter
import os

#打开一个excel文件
def open_xls(file):
    fh=xlrd.open_workbook(file)
    return fh

#获取excel中所有的sheet表
def getsheet(fh):
    return fh.sheets()

#获取sheet表的行数
def getnrows(fh,sheet):
    table=fh.sheets()[sheet]
    return table.nrows

#读取文件内容并返回行内容
def getFilect(file,shnum):
    fh=open_xls(file)
    table=fh.sheets()[shnum]
    num=table.nrows
    for row in range(num):
        rdata=table.row_values(row)
        datavalue.append(rdata)
    return datavalue

#获取sheet表的个数
def getshnum(fh):
    x=0
    sh=getsheet(fh)
    for sheet in sh:
        x+=1
    return x


if __name__=='__main__':
    #定义要合并的excel文件列表
    #allxls=['/Users/sunhaofei/Downloads/rename_text/2/1.xlsx','/Users/sunhaofei/Downloads/rename_text/2/2.xlsx']
    

    dir = '/Users/sunhaofei/Downloads/rename_text/2/'
    allxls = os.listdir(dir)
    
    #存储所有读取的结果
    datavalue=[]
    for fl in allxls:
        a= os.path.join(dir, fl)
        fh=open_xls(a)
        x=getshnum(fh)
        for shnum in range(x):
            print("正在读取文件："+str(a)+"的第"+str(shnum)+"个sheet表的内容...")
            rvalue=getFilect(a,shnum)
    #定义最终合并后生成的新文件
    endfile='/Users/sunhaofei/Downloads/rename_text/2/11.xlsx'
    wb1=xlsxwriter.Workbook(endfile)
    #创建一个sheet工作对象
    ws=wb1.add_worksheet()
    for a in range(len(rvalue)):
        for b in range(len(rvalue[a])):
            c=rvalue[a][b]
            ws.write(a,b,c)
    wb1.close()
    print("文件合并完成")
