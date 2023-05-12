#数据集划分程序，current_path为derm数据集的train或test数据路径，new_path为分类后路径，分类后使用deldoc.py进行小类别删除
import os
import shutil

def deldigit(string):
    newstring = ''.join([i for i in string if not i.isdigit()])
    newstring = newstring.lower()
    #print(newstring)
    return newstring
current_path = os.getcwd() + "/"+"test"
new_path = os.getcwd() + "/"+"mytest"
#print('当前目录：'+current_path)

document_list = os.listdir(current_path)
for document in document_list:
    if document == "mytest":
        continue
    print('当前目录：'+current_path + "/" + document)
    filename_list = os.listdir(current_path + "/" + document)
    print('当前目录下文件：',filename_list)

    print('正在分类整理进文件夹ing...')
    for filename in filename_list:
        name1, name2 = filename.split('.')
        name3 = deldigit(name1)
        if name2 == 'jpg' or name2 == 'png':
            try:
                os.mkdir(new_path + "/"+ name3[:-1])
                print('创建文件夹'+name3[:-1])
            except:
                pass
            try:
                shutil.copy(current_path+'/'+document+'/'+filename, new_path+'/'+ name3[:-1])
                print(filename+'转移成功！')
                pass
            except Exception as e:
                print('移动失败:' + e)


print('整理完毕！')
#input()
