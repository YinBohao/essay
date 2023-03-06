import torch
import numpy as np
import time

X = torch.randn(20000, 20000)
Y = torch.randn(20000, 20000)

start1 = time.time()
# numpy内积
Z = np.dot(X, Y)
end1 = time.time()
print (end1-start1)

start2 = time.time()
# torch内积
Z = X.mm(Y)
end2 = time.time()
print (end2-start2)

# print('torch ',torch.__version__)

# print('cuda ',torch.cuda.is_available(),torch.version.cuda)

# print('cudnn ',torch.backends.cudnn.is_available()
#       ,torch.backends.cudnn.version())

# print(torch.cuda.get_arch_list())
