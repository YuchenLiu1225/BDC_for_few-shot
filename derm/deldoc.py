#用于删除图片较少的类别，需要修改：第五行new_path为划分后数据集路径，第13行"mytest"改成数据集文件夹名（好像是这样），第18行的最小图片个数（20）
import os
import shutil

new_path = os.getcwd() + "/"+"mytest"
#print('当前目录：'+current_path)

document_list = os.listdir(new_path)
classes = 0
img_num = 0
max_num = 0
for document in document_list:
    if document == "mytest":
        continue
    print('当前目录：'+new_path + "/" + document)
    filename_list = os.listdir(new_path + "/" + document)
    print(len(filename_list))
    if len(filename_list) < 20:
        shutil.rmtree(new_path + "/" + document)
    else:
        img_num = img_num + len(filename_list)
        if len(filename_list) > max_num:
            max_num = len(filename_list)
        classes += 1
    #print('当前目录下文件：',filename_list)


print('整理完毕！')
print("共" + str(classes) + "类，合计" + str(img_num) + "张图片。")
print(max_num)