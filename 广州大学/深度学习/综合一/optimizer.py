import sys
import os
os.chdir(sys.path[0])

import torch

"""
定义了五个优化算法的类：SGD、MomentumSGD、AdaGrad、RMSprop和Adam。
每个类都有__init__、step和zero_grad方法。

__init__方法用于初始化优化器的参数，如学习率、动量等。
step方法用于执行一步优化，更新模型参数。
zero_grad方法用于将模型参数的梯度置零。
"""

class SGD:
    def __init__(self, parameters, lr=0.001):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()



class MomentumSGD:
    def __init__(self, parameters, lr=0.001, momentum=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(param) for param in self.parameters]

    def step(self):
        for param, velocity in zip(self.parameters, self.velocities):
            velocity.mul_(self.momentum).sub_(self.lr * param.grad)
            param.data += velocity
            # velocity.data = self.momentum * velocity + param.grad
            # param.data -= self.lr * velocity

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()


class AdaGrad:
    def __init__(self, parameters, lr=0.001, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.eps = eps
        self.accumulators = [torch.zeros_like(param) for param in self.parameters]

    def step(self):
        for param, accumulator in zip(self.parameters, self.accumulators):
            accumulator.add_(param.grad.pow(2))
            param.data -= self.lr * param.grad / (accumulator.sqrt() + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

class RMSprop:
    def __init__(self, parameters, lr=0.001, alpha=0.9, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.accumulators = [torch.zeros_like(param) for param in self.parameters]

    def step(self):
        for param, accumulator in zip(self.parameters, self.accumulators):
            accumulator.mul_(self.alpha).addcmul_(1 - self.alpha, param.grad, param.grad)
            param.data -= self.lr * param.grad / (accumulator.sqrt() + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.steps = 0
        self.momentums = [torch.zeros_like(param) for param in self.parameters]
        self.accumulators = [torch.zeros_like(param) for param in self.parameters]

    def step(self):
        self.steps += 1
        for param, momentum, accumulator in zip(self.parameters, self.momentums, self.accumulators):
            momentum.mul_(self.beta1).add_(param.grad, alpha=1 - self.beta1)
            accumulator.mul_(self.beta2).addcmul_(param.grad, param.grad, value=1 - self.beta2)
            bias_correction1 = 1 - self.beta1 ** self.steps
            bias_correction2 = 1 - self.beta2 ** self.steps
            step_size = self.lr * (torch.tensor(bias_correction2).sqrt() / bias_correction1)
            param.data -= step_size * momentum / (accumulator.sqrt() + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()



