import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 这里是创建一个数据
vegetables = ["Relation intensity"]
farmers = ["1", "2", "3", '4', '5', '6', '7', '8', '9', '10', '11', 
'12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

harvest = np.array([[  0.6575, -0.0053,  0.3726,  0.2207,  0.0809, -0.1625,  0.6614,  0.8008,
        -1.0000, -0.7511, -0.9140, -0.5053,  0.4469, -0.1645, -0.1163,  0.1985,
         0.9789,  0.5080,  0.9725,  0.1636,  1.0000,  0.4940,  0.3573]])
harvest = harvest.reshape(1, -1)

# 这里是创建一个画布
fig, ax = plt.subplots()
im = ax.imshow(harvest)

# 这里是修改标签
# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# 因为x轴的标签太长了，需要旋转一下，更加好看
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# 添加每个热力块的具体数值
# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = 1#ax.text(j, i, harvest[i, j],
               #        ha="center", va="center", color="w")
ax.set_title("RelationMap test")
fig.tight_layout()
plt.colorbar(im)
plt.show()
plt.savefig('./img/pic.png')