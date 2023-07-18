import sys
import os
os.chdir(sys.path[0])

import torch

class CrossEntropyLoss:
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, logits, targets):
        # 计算softmax预测概率
        probs = self.softmax(logits)
        # 对每个样本计算交叉熵损失
        loss = torch.mean(-torch.log(probs[range(len(targets)), targets]))
        return loss

class Softmax:
    def __call__(self, logits):
        # 获取logits中的最大值
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        # 按行减去最大值，用于数值稳定性
        logits_exp = torch.exp(logits - max_logits)
        # 计算每个样本的softmax概率
        softmax_probs = logits_exp / torch.sum(logits_exp, dim=1, keepdim=True)
        return softmax_probs

# 封装CrossEntropyLoss类
class LossFunction:
    def __init__(self):
        self.loss = CrossEntropyLoss()

    def __call__(self, logits, targets):
        return self.loss(logits, targets)

