import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import jieba

import re
import os, sys
os.chdir(sys.path[0])

# 设置随机种子
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取多个文本文件
# file_paths = glob.glob('鬼吹灯*.txt')  # 匹配开头的所有txt文件
file_paths = [
            # '鬼吹灯之精绝古城.txt', 
            # '鬼吹灯之云南虫谷.txt',
            '鬼吹灯之云南虫谷.txt'
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
# merged_data = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]+", " ", merged_data)
merged_data = jieba.lcut(merged_data)
# merged_data = [word for word in merged_data if word.isalnum()]
# merged_data = [word.lower() for word in merged_data]
# merged_data = ['<SPACE>' if word == ' ' else word for word in merged_data]



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
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        output = self.fc(output[:, -1, :])  # 只使用最后一个时间步的输出
        return output

# class AttentionGRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(AttentionGRUModel, self).__init__()
#         self.embedding = nn.Embedding(input_size, embedding_size)
#         self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
        
#     def forward(self, x):
#         embedded = self.embedding(x)
#         output, _ = self.gru(embedded)
#         output = self.fc(output[:, -1, :])  # 只使用最后一个时间步的输出
#         return output

# 设置超参数
input_size = len(word_to_index)
embedding_size = 128
hidden_size = 256
output_size = len(word_to_index)
num_layers = 2
learning_rate = 0.001
num_epochs =20

# 创建模型实例
model = GRUModel(input_size, hidden_size, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 计算困惑度
def calculate_perplexity(loss):
    return torch.exp(loss)

# 定义空列表存储训练过程中的损失值
losses = []
# 定义空列表存储训练过程中的困惑度值
perplexities = []
# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for inputs_batch, targets_batch in dataloader:
        inputs_batch = inputs_batch.to(device)
        targets_batch = targets_batch.to(device)
        
        # 前向传播
        outputs = model(inputs_batch)
        loss = criterion(outputs, targets_batch)
        
        total_loss += loss.item()
        num_batches += 1
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 计算平均损失和困惑度
    average_loss = total_loss / num_batches
    perplexity = calculate_perplexity(average_loss)
    
    # 打印训练过程中的损失和困惑度
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}, Perplexity: {perplexity}")

    # 记录困惑度值
    perplexities.append(perplexity.item())  
    # 记录损失值
    losses.append(loss.item())

# x 轴刻度为每个 epoch 的索引值
x = range(1, len(perplexities) + 1)

# 绘制困惑度曲线
plt.plot(x,perplexities)
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity Curve')
plt.savefig('Perplexity_Curve.png', dpi=300)
plt.close()
# plt.show()

# 绘制损失曲线
# plt.plot(x,losses)
# plt.xticks(x)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.savefig('Training_Loss.png', dpi=300)
# plt.close()
# plt.show()

# 保存模型
torch.save(model.state_dict(), "model.pt")

# 文本生成
def generate_text(model, start_text, num_words):
    model.eval()
    current_text = start_text
    generated_text = start_text

    # 将初始文本转换为索引序列
    for _ in range(num_words):
        # input_indices = [word_to_index[word] for word in current_text]
        input_indices = [word_to_index[word] for word in current_text if word in word_to_index]
        if len(input_indices) == 0:
            break  # 如果当前文本中的所有词都不在字典中，则停止生成
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
        
        # 前向传播
        output = model(input_tensor)
        _, predicted_index = torch.max(output, dim=1)
        
        # 将预测的词添加到生成的文本中
        predicted_word = index_to_word[predicted_index.item()]
        generated_text += ' ' + predicted_word
        
        # 更新当前文本
        current_text = current_text[1:] + [predicted_word]
    
    return generated_text

# 加载已保存的模型
loaded_model = GRUModel(input_size, hidden_size, output_size).to(device)
loaded_model.load_state_dict(torch.load("model.pt"))

# 生成文本示例
# start_text = ['胡']
# 读取起始文本
start_text_file = 'start_text.txt'
with open(start_text_file, 'r', encoding='utf-8') as file:
    start_text = file.read().strip()
    start_text = jieba.lcut(start_text)

print(start_text)
generated_text = generate_text(loaded_model, start_text, num_words=100)

# 将生成的文本连成一段话
# print(generated_text)
generated_text=[x.strip() for x in generated_text if x.strip() != '']
generated_text = ''.join(generated_text)
# generated_text = re.sub(r'([,.!?，。！？])', r' \1 ', generated_text)


# 保存生成的文本到txt文件
output_file = "result.txt"
with open(output_file, "w", encoding="GB18030") as file:
    file.write(generated_text)

print("生成的文本已保存到文件:", output_file)

