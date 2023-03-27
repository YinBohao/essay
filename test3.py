import torch

input_dim = 10
output_dim = 10

linear = torch.nn.Linear(3, 5)

input = torch.randn(2,3)

output = linear(input)

print(input)
print(linear)
print(output)