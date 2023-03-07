import torch
import numpy as np
import time

X = torch.randn(20000, 20000)
Y = torch.randn(20000, 20000)
# start1 = time.time()
# # numpy内积
# Z = np.dot(X, Y)
# end1 = time.time()
# print (end1-start1)

device = torch.device('cuda:0' )
X = X.to(device)
Y = Y.to(device)

start2 = time.time()
# torch内积
Z = X.mm(Y)
print(Z.cuda().device)
end2 = time.time()
print (end2-start2)

# print('torch ',torch.__version__)
# print(torch.cuda.device_count())

# print('cuda ',torch.cuda.is_available(),torch.version.cuda)

# print('cudnn ',torch.backends.cudnn.is_available()
#       ,torch.backends.cudnn.version())
# print(torch.cuda.get_device_name(0))         # 查看使用的设备名称

# print(torch.cuda.get_arch_list())
# import torch
# print(torch.cuda.get_device_name(0))         # 查看使用的设备名称
# print(torch.cuda.is_available())	     # 验证cuda是否正常安装并能够使用
# ng = torch.cuda.device_count()
# print("Devices:%d" %ng)
# import torch
# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' #使用多块显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,'      #只使用编号为0的显卡

# A=torch.arange(12,dtype=torch.float32)
# B=A.cuda()
# print(B.device)
# print("A=\n",A)