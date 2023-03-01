import torch
import torchvision.models as models
from Utils import load

model = load.model('vgg16', 'lottery')((3, 32, 32), 10, False, False)
device = torch.device('cpu')
model.load_state_dict(torch.load('/scratch/network/ogolev/Results/data/singleshot/hp-rand-0.5/model.pt', map_location=device))
i = 0
for layer in model.children():
    print(layer.state_dict()[str(i) + '.conv.weight'])
    i += 1
