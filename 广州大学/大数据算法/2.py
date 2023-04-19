import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import os
import sys
os.chdir(sys.path[0])

import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义转换方法
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),  # 中心裁剪
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5,), std=(0.5,)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化处理
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
trainset = datasets.ImageFolder('GTSRB/train', transform=transform)
testset = datasets.ImageFolder('GTSRB/test', transform=transform)

# 加载数据
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 3. 定义一个函数来迭代DataLoader对象并将数据展平为一维张量
def flatten_data(loader):
    images_list = []
    labels_list = []

    for images, labels in loader:
        # 将数据展平为一维张量
        images = images.view(images.size(0), -1)
        images_list.append(images)

        # 将标签添加到标签列表中
        labels_list.append(labels)

    # 将多个张量组合成一个大张量
    images_concat = torch.cat(images_list, dim=0)
    labels_concat = torch.cat(labels_list, dim=0)

    return images_concat , labels_concat

# 4. 调用该函数来加载并展平训练集和测试集中的图像和标签
train_images, train_labels = flatten_data(trainloader)
test_images, test_labels = flatten_data(testloader)

# 5. 将数据用列表存储
train_images_list = train_images.cpu().detach().numpy().tolist()
test_images_list = test_images.cpu().detach().numpy().tolist()
train_labels_list = train_labels.cpu().detach().numpy().tolist()
test_labels_list = test_labels.cpu().detach().numpy().tolist()


from sklearn import svm

# 新建SVM分类器
svm_clf = svm.SVC(kernel='linear')

# 使用训练集数据训练SVM分类器
svm_clf.fit(train_images_list, train_labels_list)
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import accuracy_score
# # 定义参数搜索空间
# parameters = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}

# # 使用 APO 算法搜索最佳的超参数组合  
# clf_apo = SVC(kernel='linear', C=1, random_state=42)  
# clf_apo = GridSearchCV(clf_apo, parameters, cv=5)  
# clf_apo.fit(train_images_list, train_labels_list)

# # 输出最佳的超参数组合
# print("Best Parameters: ", clf_apo.best_params_)

# # 训练模型
# clf = SVC(kernel='linear', C=1, random_state=42)
# clf.set_params(**clf_apo.best_params_)
# clf.fit(train_images_list, train_labels_list)

# # 在测试集上进行评估

# predictions = clf.predict(test_images_list)
# accuracy = accuracy_score(test_labels_list, predictions)
# print("Accuracy: ", accuracy)
# print('test_accuracy: {:.4f}%'.format(accuracy*100.))

# 使用测试集数据评估SVM分类器的性能
accuracy = svm_clf.score(test_images_list, test_labels_list)
print('test_accuracy: {:.4f}%'.format(accuracy*100.))

# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score


# # 训练 SVM 模型  
# clf = SVC(kernel='linear', C=1, random_state=42)  
# clf.fit(train_data, train_labels)

# # 使用交叉验证评估模型的性能  
# scores = cross_val_score(clf, test_data, test_labels, cv=5)  
# print('Cross-validation scores:', scores)  





# # 使用自动参数优化算法来选择最佳的超参数组合  
# # best_score = -1  
# # best_params = None  
# # for i in range(10):  
# #     # 使用不同的超参数组合进行训练和评估  
# #     clf_params = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'random_state': [0, 42]}  
# #     clf_apo = SVC(**clf_params)  
# #     clf_apo.fit(train_data, train_labels)  
# #     score = cross_val_score(clf_apo, test_data, test_labels, cv=5)  
# #     if score > best_score:  
# #         best_score = score  
# #         best_params = clf_params  
# # print('Best score:', best_score)