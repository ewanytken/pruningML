import copy
from typing import Optional, Union, Any, List, Tuple
import torch
import numpy as np
import torch.cuda
from datasets import DatasetDict
from data.LoggerWrapper import LoggerWrapper

logger = LoggerWrapper()

class RationalWeight:

    model: Union[Any, None] = None
    dataset: Union[Optional[DatasetDict], None] = None
    tokenizer: Union[Any, None] = None

    def __init__(self, model, tokenizer, dataset, device="cpu"):

        if torch.cuda.is_available():
            device = "cuda"
        modelCopy = copy.deepcopy(model)
        self.model = modelCopy.to(device)
        self.tokenizer = tokenizer
        self.dataset = dataset

        assert model is not None, "Exception: NO MODEL"
        assert dataset is not None, "Exception: NO DATASET"
        assert tokenizer is not None, "Exception: NO TOKENIZER"

    # freeze weight, except specify layers
    def weightFreeze(self, layers: Union[List[str], str] = None) -> None:

        if isinstance(layers, str):
            layers = []
            layers.append(layers)

        for name, param in self.model.named_parameters():
            # logger(name)
            param.requires_grad = False

        if layers is not None:
            for name, param in self.model.named_parameters():
                for layer in layers:
                    if layer in name:
                        logger(f'''\nUnfreeze parameters: {name}''')
                        param.requires_grad = True

    # compute 3 sigma interval
    def threeSigmaCompute(self, layer: str, deflate: int = 1) -> Tuple[List[int | Any], List[int | Any], List[int | Any]]:
        assert isinstance(layer, str), "Exception: Layer must be String type"

        weightTensor = self.layerSearch(layer)

        flatWeight = weightTensor.reshape(1, -1)[0].cpu().detach().numpy()

        one =   [flatWeight.mean() - 1 * flatWeight.std()/deflate,
                 flatWeight.mean() + 1 * flatWeight.std()/deflate]

        two =   [flatWeight.mean() - 2 * flatWeight.std()/deflate,
                 flatWeight.mean() + 2 * flatWeight.std()/deflate]

        three = [flatWeight.mean() - 3 * flatWeight.std()/deflate,
                 flatWeight.mean() + 3 * flatWeight.std()/deflate]

        # logger(f" \n One sigma: {one}, \n Two sigma: {two}, \n Three sigma: {three}")
        return one, two, three

    # Transforms weight in model
    def weightRationalization(self, layer: str, interval: list,
                              scale: float = 1, inside: bool = True) -> None:

        assert isinstance(layer, str), "Exception: Layer must be String type"

        param = self.layerSearch(layer)
        # logger(f"Basic parameters: {param} \n")

        if inside is True:   # set to zero inside distribution
            # Define two conditions
            condition1 = param >= interval[0] # 0 -> negative
            condition2 = param <= interval[1] # 1 -> positive
            combined_condition = condition1 & condition2

        else:                # set to zero outside distribution
            condition1 = param <= interval[0] # 0 -> negative
            condition2 = param >= interval[1] # 1 -> positive
            combined_condition = condition1 | condition2

        transformed_param = torch.where(combined_condition, 0, param)
        transformed_param = torch.where(param != 0, transformed_param * scale, transformed_param)

        self.layerSearch(layer).copy_(transformed_param)

        countZero = torch.sum(transformed_param == 0)
        print(f'''\nNumber of ZERO: {countZero};\nPercent of ZERO: {countZero / torch.numel(transformed_param)};
        \n''')

        # logger(f"New parameters: {self.layerSearch(layer)}")

    # compute data for metrics calculation
    def outputCompute(self, numberOfsampling: int = 40,
                            max_length: int = 512,
                            datasetName: str = 'test') -> Tuple[np.ndarray, np.ndarray,np.ndarray]:

        probList = []
        labelList = []
        indexList = []

        for i, prompt in enumerate(self.dataset[datasetName]['text'][:numberOfsampling]):
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = inputs["input_ids"]

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            outputs = self.model(inputs)
            softmax = torch.nn.Softmax(dim=1)
            outSoftmax = softmax(outputs["logits"]).cpu()
            outIndex = torch.argmax(outSoftmax)
            outIndex = outIndex

            label = self.dataset[datasetName]['label'][i]

            probList.append(outSoftmax[0].tolist())
            indexList.append(outIndex.item())
            labelList.append(label)

            # logger(f"\nLabel: {label} \nProb: {outSoftmax} \nIndexMax: {outIndex}")

        return np.array(labelList), np.array(probList), np.array(indexList)

    def layerSearch(self, layer: str) -> torch.Tensor:
        for name, param in self.model.named_parameters():
            if layer in name:
                return param
        assert isinstance(layer, torch.Tensor), f'''Don't find any layer with name: {layer}'''

    def weightLayersRationalization(self, layers: list, sigma: int, scale: float, inside: bool) -> None:
        assert (sigma >= 1 & sigma <= 3), "Error: Sigma value exceed from interval"
        assert layers.__len__() != 0, "Error: Empty list obtain"
        for layer in layers:
            interval = [self.threeSigmaCompute(layer)][0]
            self.weightRationalization(layer, interval[sigma-1], scale, inside)

    def weightRationalizationWithoutModule(self, nameExclude: list, sigma: int, scale: float, inside: bool) -> List[Any]:
        assert (sigma >= 1 & sigma <= 3), "Error: Sigma value exceed from interval"

        exclude = []
        for name, param in self.model.named_parameters():
            for excludeName in nameExclude:
                if excludeName in name:
                    exclude.append(name)

        layers = list(filter(lambda x: x not in exclude, [name for name, param in self.model.named_parameters()]))

        for layer in layers:
            interval = [self.threeSigmaCompute(layer)][0]
            self.weightRationalization(layer, interval[sigma-1], scale, inside)

        return layers
