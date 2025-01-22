import copy
from typing import Optional, Union, Any, List, Tuple
import torch
import numpy as np
import torch.cuda
from datasets import DatasetDict
from data.LoggerWrapper import LoggerWrapper
from data import RationalWeight
from data import binaryMetricsCompute
from data import metricsIndicator

logger = LoggerWrapper()

class Enumerator:

    rationalWeight: Union[RationalWeight, None] = None

    def __init__(self, rationalWeight: RationalWeight):
        self.rationalWeight = rationalWeight
        assert rationalWeight is not None, "Exception: NO RationalWeight object"

    def computeMetricsValue(self, layers: list, sigma: list,
                            scale: list, inside: bool, numberOfSample: int) -> list:

        self.rationalWeight.weightFreeze()

        result = []

        for sc in scale:
            for s in sigma:
                tempRationalWeight = copy.deepcopy(self.rationalWeight)
                tempRationalWeight.weightLayersRationalization(layers, s, sc, inside)
                label, soft, pred = tempRationalWeight.outputCompute(numberOfSample)
                dictOfMetrics = binaryMetricsCompute(label, pred, soft)
                dictParam = {"sigma:" : s, "scale:": sc, "inside:" : inside, "RESULT:":dictOfMetrics}
                result.append(dictParam)
        return result

    def combainPruning(self, layers: list,
                             sigma: list,
                             scale: list,
                             inside: bool,
                             numberOfSample: int,
                             decrease: float) -> list:

        result = []
        self.rationalWeight.weightFreeze()
        label, soft, pred = self.rationalWeight.outputCompute(numberOfSample)
        metrics = metricsIndicator(label, pred, soft)

        for s in sigma:
            for sc in scale:

                stateRationalWeight = copy.deepcopy(self.rationalWeight)
                newStatePrun = copy.deepcopy(self.rationalWeight)

                fullData = None
                layersWithPrun = []

                for layer in layers:
                    interval = [self.rationalWeight.threeSigmaCompute(layer)][0]

                    stateRationalWeight.weightRationalization(layer, interval[s-1], sc, inside)
                    label, soft, pred = stateRationalWeight.outputCompute(numberOfSample)
                    prunMetrics = metricsIndicator(label, pred, soft)

                    logger(metrics)
                    logger(prunMetrics)

                    if (metrics["Accuracy"] * decrease < prunMetrics["Accuracy"] ) \
                            & (metrics["ROC_AUC_score"] * decrease < prunMetrics["ROC_AUC_score"]) \
                            & (metrics["F1"] * decrease < prunMetrics["F1"]) \
                            & (metrics["Precision"] * decrease < prunMetrics["Precision"]) \
                            & (metrics["Recall"] * decrease < prunMetrics["Recall"]):
                        layersWithPrun.append(layer)
                        fullData = { "sigma": s,
                                     "scale": sc,
                                     "inside": inside,
                                     "RESULT": prunMetrics,
                                     "Pruning_layers": layersWithPrun}
                        newStatePrun = copy.deepcopy(stateRationalWeight)
                    else:
                        stateRationalWeight = copy.deepcopy(newStatePrun)

                result.append({"prunModel": stateRationalWeight,
                               "data": fullData})

        return result
