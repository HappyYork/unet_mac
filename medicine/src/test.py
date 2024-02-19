import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

test = np.arange(11)
input = torch.tensor(np.array([test[i:(i + 3)] for i in range(10 - 1)]))
target = torch.tensor(np.array([test[i:(i + 1)] for i in range(10 - 1)]))


torch_dataset = TensorDataset(input, target)

def collate(data):
    #result = lambda data: data
    img = []
    label = []
    for each in data:
        img.append(each[0])
        label.append(each[1])
    return img,label

my_dataloader = DataLoader(
    dataset=torch_dataset,
    batch_size=4,
    # collate_fn=lambda x: x
    collate_fn = collate
)
for i in my_dataloader:
    print('*' * 30)
    print(i)
