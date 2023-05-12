import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from PIL import Image
import numpy as np
 
if __name__ == '__main__':
    #预处理操作
    to_tensor=transforms.Compose([transforms.Resize((256,256),2),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #load model
    model=models.resnet101(pretrained=False)
    load_checkpoint="./resnet101-63fe2227.pth"
    state_dict = torch.load(load_checkpoint,map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    #../Derm/new/atypical-nevi/atypical-nevi-19.jpg
    img = Image.open("../Derm/new/atypical-nevi/atypical-nevi-19.jpg").convert("RGB")
    #img = Image.open("../Derm/new/acanthosis-nigricans/acanthosis-nigricans-1.jpg").convert("RGB")
    print(list(model.children()))
    features=list(model.children())[:-1]#去掉池化层及全连接层
    #print(list(model.children())[:-2])
    modelout=nn.Sequential(*features).to(device)
    
    img_tensor=to_tensor(img).unsqueeze(0).to(device,torch.float)
    print(img_tensor.shape)
    out=modelout(img_tensor)
    #print(out)
    out = out.cpu().detach().numpy()
    print(out.shape)
    #np.save('test.npy', out)
    #te = np.load('test.npy')
    #print(te.shape)
