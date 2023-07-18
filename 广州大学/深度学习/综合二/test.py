import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import jieba
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
os.chdir(sys.path[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 文本生成模型


class AttentionGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)

        attention_weights = torch.softmax(
            self.attention(output).squeeze(2), dim=1)
        attended_output = torch.bmm(
            attention_weights.unsqueeze(1), output).squeeze(1)

        output = self.fc(attended_output)
        return output

# 计算困惑度


def calculate_perplexity(loss):
    loss_tensor = torch.tensor(loss, dtype=torch.float32)
    return torch.exp(loss_tensor)


# 文本生成
def generate_text(model, start_text, num_words=100, temperature=0.8, beam_width=5):
    model.eval()
    hidden = None

    # 将起始文本转换为张量
    input_tensor = torch.tensor(
        [word_to_index[word] for word in start_text], dtype=torch.long).unsqueeze(0).to(device)

    # 生成文本
    generated_words = start_text[:]
    for _ in range(num_words):
        output = model(input_tensor)
        output = output.squeeze(0) / temperature
        probabilities = F.softmax(output, dim=-1).cpu().detach().numpy()

        # 使用束搜索获取下一个词的候选
        top_indexes = get_top_indexes(probabilities, beam_width)
        next_indexes = random.choice(top_indexes)

        # 更新生成的文本
        next_word = index_to_word[next_indexes]
        generated_words.append(next_word)

        # 更新输入文本
        input_tensor = torch.tensor(
            [next_indexes], dtype=torch.long).unsqueeze(0).to(device)

    return generated_words


def get_top_indexes(probabilities, beam_width):
    top_indexes = []
    prob_sorted = sorted(probabilities, reverse=True)
    for i in range(beam_width):
        index = np.where(probabilities == prob_sorted[i])[0][0]
        top_indexes.append(index)
    return top_indexes


# 读取多个文本文件
file_paths = [
    # '鬼吹灯之精绝古城.txt',
    '鬼吹灯之龙岭迷窟.txt',
    # '鬼吹灯之云南虫谷.txt'
]
data = []
for file_path in file_paths:
    with open(file_path, 'r', encoding='GB18030') as file:
        file_data = file.read()
        data.append(file_data)

# 合并文本数据
merged_data = ' '.join(data)

# 文本预处理
merged_data = merged_data.strip()
merged_data = merged_data.replace('\n', '')  # 移除换行符
merged_data = merged_data.replace(' ', '')   # 移除空格
merged_data = jieba.lcut(merged_data)
# merged_data = [word for word in merged_data if word.isalnum()]
# merged_data = [word.lower() for word in merged_data]

# 创建字典
word_to_index = {}
index_to_word = {}
for word in merged_data:
    if word not in word_to_index:
        index = len(word_to_index)
        word_to_index[word] = index
        index_to_word[index] = word

# 将文本转换为索引序列
data_indices = [word_to_index[word] for word in merged_data]

# 生成输入序列和目标序列
sequence_length = 20  # 输入序列长度
inputs = []
targets = []
for i in range(len(data_indices) - sequence_length):
    inputs.append(data_indices[i:i+sequence_length])
    targets.append(data_indices[i+sequence_length])

# 转换为张量
inputs = torch.tensor(inputs, dtype=torch.long).to(device)
targets = torch.tensor(targets, dtype=torch.long).to(device)

# 创建数据加载器
dataset = TensorDataset(inputs, targets)
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型和优化器初始化
input_size = len(word_to_index)
embedding_size = 128
hidden_size = 256
output_size = len(word_to_index)
num_layers = 2
# 创建模型实例
model = AttentionGRUModel(input_size, hidden_size, output_size).to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
perplexities = []  # 记录困惑度变化
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_inputs, batch_targets in dataloader:
        batch_inputs, batch_targets = batch_inputs.to(
            device), batch_targets.to(device)
        # 前向传播
        output = model(batch_inputs)

        # 计算损失
        loss = criterion(output, batch_targets)

        total_loss += loss.item()
        num_batches += 1

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 计算平均损失和困惑度
    average_loss = total_loss / num_batches
    perplexity = calculate_perplexity(average_loss)
    perplexities.append(perplexity.item())

    # 打印训练过程中的损失和困惑度
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}, Perplexity: {perplexity}")

# x 轴刻度为每个 epoch 的索引值
x = range(1, len(perplexities) + 1)
# 绘制困惑度曲线
plt.plot(x, perplexities)
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity Curve')
plt.savefig('Perplexity_Curve.png', dpi=300)
plt.close()
# plt.show()

# 保存模型
torch.save(model.state_dict(), "model.pt")

# 加载已保存的模型
loaded_model = AttentionGRUModel(
    input_size, hidden_size, output_size).to(device)
loaded_model.load_state_dict(torch.load("model.pt"))

# 读取起始文本
start_text_file = 'start_text.txt'
with open(start_text_file, 'r', encoding='utf-8') as file:
    start_text = file.read().strip()
    start_text = jieba.lcut(start_text)

print(start_text)
generated_text = generate_text(loaded_model, start_text, num_words=100)

# 将生成的文本连成一段话
print(generated_text)
generated_text = [x.strip() for x in generated_text if x.strip() != '']
generated_text = ''.join(generated_text)
# generated_text = re.sub(r'([,.!?，。！？])', r' \1 ', generated_text)

# 保存生成的文本到txt文件
output_file = "result.txt"
with open(output_file, "w", encoding="GB18030") as file:
    file.write(generated_text)

print("生成的文本已保存到文件:", output_file)
