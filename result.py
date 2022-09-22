import torch
import numpy as np

from plyfile import PlyData, PlyElement


import torch
import numpy as np
resultPath = "/qys/cuda10docker/BPNet-main/Data/ScanNet/"

# pred = np.load(resultPath + "result/best/pred.npy")
# gt = np.load(resultPath + "result/best/gt.npy")
# # ply = np.load()
# print("0")
def float2color(zero2one):
    x = zero2one * 256 * 256 * 256
    r = x % 256
    g = ((x - r)/256 % 256)
    b = ((x - r - g * 256)/(256*256) % 256)
    r = round(float(r/256),2)
    g = round(float(g/256),2)
    b = round(float(b/256),2)
    return [r,g,b]

def Color_To_RGB(color):
    
    b = color / (256 * 256)
    g = (color - b * 256 * 256) / 256
    r = color - b * 256 * 256 - g * 256
    return [r,g,b]


colordict = {
    0:[230,107,26],
    1:[213,179,43],
    2:[107,230,26],
    3:[51,204,204],
    4:[17,194,238],
    5:[26,107,230],
    6:[34,72,221],
    7:[107,26,230],
    8:[114,60,196],
    9:[140,68,187],
    10:[204,51,174],
    11:[170,85,119],
    12:[213,43,77],
    13:[130,136,119],
    14:[190,226,235],
    15:[181,244,194],
    16:[224,201,215],
    17:[244,181,181],
    18:[227,221,198],
    19:[230,196,202],
    20:[255,170,238],
    255:[255,255,170]    
}


# # 顏色值
# for i in pred:
#     rgb = colordict[i]

# pth = np.load(resultPath+"train/scene0241_00_vh_clean_2.pth")


# print("0")




def main():
    
    pth = torch.load("scene0241_00_vh_clean_2.pth")
    points = torch.tensor(pth[0])
    color = torch.tensor(pth[1])
    gt = pth[2]
    pred = np.load("./pred.npy")
    
    pred_colors = []
    for ind in range(len(pred)):
        pred_colors.append(colordict[pred[ind]])
    
    # for i in pred:
    #     if  i!=2:
    #         print(i)
    # pred =torch.t( torch.Tensor(pred))
    matrix =  torch.cat((points,torch.Tensor(pred_colors)),dim=1)
    np.savetxt(r'A.txt', matrix, fmt='%f', delimiter=',')
    print(0);

if __name__ == '__main__':
    main()