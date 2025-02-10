import torch
from torch import topk #return max element in dim (can return K-max elements)
import numpy as np

# Hook facility recall every time after forward function will be call
class Hook():
    features = []
    def __init__(self, m, grad_true = False):
        if grad_true is False:
            self.hook = m.register_forward_hook(self.hookFunc)
        else:
            self.hook = m.register_backward_hook(self.hookFunc)
    def hookFunc(self, module, input, output):
        self.features.append(output.clone().detach())
    def remove(self):
        self.hook.remove()
    def clearFeatures(self):
        self.features.clear()

# Method return list of activision values
def actLayerMethod(model: torch.nn.Module, preprocessingList: list, \
                   layer: str = 'layer4') -> list:
    hookForLayer = Hook(model._modules.get(layer))
    predictionList = [model(pred) for pred in preprocessingList]
    hookForLayer.remove()
    features = hookForLayer.features.copy()
    hookForLayer.clearFeatures()
    return features, predictionList

# Method return max idx for prediction list. Len accordance to len of dataset
def maxActivisionValue(predictionList : list) -> list:
    return [torch.squeeze(topk(pred, 1)[1].int()).numpy() \
                                         for pred in predictionList]

# Need for FC layer, but can gain weight from other layers with squeeze 2 last dimension
def weightFromLayer(model : torch.nn.Module, layer : str = 'fc') -> np.ndarray:
    return np.squeeze(list(model._modules.get(layer).parameters())[0].cpu().data.numpy())