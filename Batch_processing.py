import pandas as pd
import os

def aa(file):
    df = pd.read_excel(file)
    print(df)


def eachFile(filepath):
    pathDir = os.listdir(filepath)  # 获取当前路径下的文件名，返回List
    for s in pathDir:
        newDir = os.path.join(filepath, s)  # 将文件命加入到当前文件路径后面
        if os.path.isfile(newDir):  # 如果是文件
            if os.path.splitext(newDir)[1] == ".xlsx":  # 判断是否是xlsx
                try:
                    aa(newDir)  # 读文件
                except Exception:
                    pass
                continue
                pass
        else:
            eachFile(newDir)


eachFile("/Users/sunhaofei/Downloads/rename_text/1/")
