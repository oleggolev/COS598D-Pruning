import torch
import torchvision.models as models
from Utils import load

model = load.model('vgg16', 'lottery')((3, 32, 32), 10, False, False)
device = torch.device('cpu')
model.load_state_dict(torch.load('/scratch/network/ogolev/Results/data/singleshot/hp-rand-0.5/model.pt', map_location=device))
for i in range(16):
    try:
        conv_layer = model.layers[i].conv
        weights = conv_layer.state_dict()['weight']
        print(weights)
    except:
        # this is a MaxPool2D layer, so we don't care about this one?                                                                                                                        
        continue
