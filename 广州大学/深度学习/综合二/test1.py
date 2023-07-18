import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import matplotlib.pyplot as plt
import numpy as np
import os, sys
os.chdir(sys.path[0])

# 文本预处理和可视化函数封装
def preprocess_and_visualize(file_path):
    # 读取文本文件
    with open(file_path, 'r', encoding='GB18030', errors='ignore') as file:
        text = file.read()
    
    # 文本预处理
    text = text.replace('\n', '')  # 移除换行符
    text = text.replace(' ', '')   # 移除空格
    
    # 使用jieba分词
    words = jieba.lcut(text)
    
    # 统计词频
    word_counts = {}
    for word in words:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    
    # 可视化词频分布
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [word[0] for word in sorted_word_counts[:10]]
    top_word_counts = [word[1] for word in sorted_word_counts[:10]]
    
    plt.bar(top_words, top_word_counts)
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.title('Top 10 Words in Text')
    # plt.show()

# GRU模型定义
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, input):
        hidden = torch.zeros(1, input.size(1), self.hidden_size).to(device)
        output, _ = self.gru(input, hidden)
        output = self.fc(output)
        output = self.softmax(output)
        return output

# 文本预处理和可视化示例
preprocess_and_visualize('111.txt')

# 训练GRU模型
def train_gru_model(train_file, input_size, hidden_size, num_epochs, batch_size):
    # 读取训练集文本文件
    with open(train_file, 'r', encoding='GB18030', errors='ignore') as file:
        text = file.read()
    
    # 文本预处理
    text = text.replace('\n', '')  # 移除换行符
    text = text.replace(' ', '')   # 移除空格
    
    # 使用jieba分词
    words = jieba.lcut(text)
    
    # 构建词汇表
    vocab = sorted(set(words))
    vocab_size = len(vocab)
    word_to_index = {word: index for index, word in enumerate(vocab)}
    
    # 构建训练集数据
    input_data = []
    target_data = []
    for i in range(len(words) - input_size):
        input_sequence = words[i:i+input_size]
        target_sequence = words[i+input_size]
        input_data.append([word_to_index[word] for word in input_sequence])
        target_data.append(word_to_index[target_sequence])
    
    # 转换为Tensor并进行批次划分
    input_tensor = torch.tensor(input_data).unsqueeze(2)
    target_tensor = torch.tensor(target_data)
    dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型和优化器
    model = GRUModel(input_size, hidden_size, vocab_size).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 模型训练
    total_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        total_loss.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    # 可视化损失曲线
    plt.plot(total_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    # plt.show()

# 设置GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练GRU模型示例
train_gru_model('111.txt', input_size=5, hidden_size=128, num_epochs=100, batch_size=64)

# 使用训练好的GRU模型生成句子
def generate_sentence(model, start_sequence, length, word_to_index, index_to_word):
    input_sequence = start_sequence.copy()
    
    # 预测并生成句子
    for _ in range(length):
        input_tensor = torch.tensor([[word_to_index[word] for word in input_sequence]]).unsqueeze(2).to(device)
        output_tensor = model(input_tensor)
        _, predicted_index = torch.max(output_tensor[:, -1], dim=2)
        predicted_word = index_to_word[predicted_index.item()]
        input_sequence.append(predicted_word)
    
    generated_sentence = ''.join(input_sequence)
    return generated_sentence

# 读取训练集文本文件

with open('111.txt', 'r', encoding='GB18030', errors='ignore') as file:
    text = file.read()

# 文本预处理
text = text.replace('\n', '')  # 移除换行符
text = text.replace(' ', '')   # 移除空格

# 使用jieba分词
words = jieba.lcut(text)

# 加载词汇表
vocab = sorted(set(words))
vocab_size = len(vocab)
word_to_index = {word: index for index, word in enumerate(vocab)}
index_to_word = {index: word for index, word in enumerate(vocab)}

# 加载训练好的模型
model = GRUModel(input_size=5, hidden_size=128, output_size=vocab_size).to(device)
model.load_state_dict(torch.load('gru_model.pt'))
model.eval()

# 设置初始序列并生成句子
start_sequence = ['六天']
generated_sentence = generate_sentence(model, start_sequence, length=10, word_to_index=word_to_index, index_to_word=index_to_word)
print(generated_sentence)
