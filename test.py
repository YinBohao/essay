import torch
print('torch ',torch.__version__)

print('cuda ',torch.cuda.is_available(),torch.version.cuda)

print('cudnn ',torch.backends.cudnn.is_available()
      ,torch.backends.cudnn.version())

print(torch.cuda.get_arch_list())