import torch
import torch.nn.functional as F
import math

# 创建模型预测结果和真实标签
predictions = torch.tensor([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]])
labels = torch.tensor([2, 0])

# 定义每个类别的权重
weights = torch.tensor([1.0, 2.0, 3.0])

# 使用F.cross_entropy计算带权重的交叉熵损失
loss = F.cross_entropy(predictions, labels, weight=weights)
print(loss) # tensor(0.8773)

# 测试计算过程
pred = F.softmax(predictions, dim=1)
print(pred)
loss2 = -(3 * math.log(pred[0,2]) + math.log(pred[1,0]))/4  # 4 = 1+3 对应权重之和
print(loss2) # 0.8773049571540321
