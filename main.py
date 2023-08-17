from dataloader import flow_load
from dataloader import label_load
import torch
from model import Vit

model_param = 'model.pth'
if __name__ == '__main__':
    model = Vit()
    model.load_state_dict(torch.load(model_param))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
