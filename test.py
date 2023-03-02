import torch
print(torch.__version__)

print(torch.cuda.is_available())
print(torch.version.cuda)

print(torch.backends.cudnn.is_available())
print(torch.backends.cudnn.version())

print(torch.cuda.get_arch_list())