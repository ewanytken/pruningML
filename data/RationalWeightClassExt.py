from data.LoggerWrapper import LoggerWrapper
from data import RationalWeight
import torch

log = LoggerWrapper()

class RationalWeightExt(RationalWeight):

    def __init__(self, model: callable or None, tokenizer, dataset, device: str = 'cpu', **kwargs):
        super().__init__(model, tokenizer, dataset, device)
        self.zeroList = []

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
        self.zeroList.append({"layer": layer, "countZero": countZero, "percentZero": countZero / torch.numel(transformed_param)})

        # print(f'''\nNumber of ZERO: {countZero};\nPercent of ZERO: {countZero / torch.numel(transformed_param)};
        # \n''')
