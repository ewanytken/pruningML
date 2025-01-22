import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
from scipy import stats
from scipy.stats import logistic, norm
from torch import topk
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plotDistribution(weightTensor: torch.Tensor) -> None:

    countZero = torch.sum(weightTensor == 0)

    features = weightTensor.reshape(1, -1)[0].cpu().detach().numpy()

    print(f'''\nNumber of ZERO: {countZero};\nPercent of ZERO: {countZero / torch.numel(weightTensor)};
    \n''')

    plt.hist(features, bins='auto', density=True)
    plt.plot(np.sort(np.array(features)),
             norm.pdf(np.sort(np.array(features)),
                      features.mean(),
                      features.std()))

    plt.plot(np.sort(np.array(features)),
             logistic.pdf(np.sort(np.array(features)),
                          features.mean(),
                          features.std()))

    plt.title(
        f'\nCount of Zero in tensor: {countZero}' +

        '\nNorm statistic: ' + str(stats.kstest(np.sort(np.array(features)),
                                                norm.cdf(np.sort(np.array(features)),
                                                         features.mean(),
                                                         features.std())).statistic) +

        '\nLog statistic: ' + str(stats.kstest(np.sort(np.array(features)),
                                               logistic.cdf(np.sort(np.array(features)),
                                                            features.mean(),
                                                            features.std())).statistic), fontsize=7)


def plotRocAuc(label: np.ndarray, outIndex: np.ndarray) -> None:
    fpr, tpr, threshold = metrics.roc_curve(label, outIndex)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('ROC AUC')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plotConfusionMatrix(label: np.ndarray, outIndex: np.ndarray) -> None:
    conf_matrix = confusion_matrix(label, outIndex)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def maxActivisionValue(predictionList: list) -> list:
    return [torch.squeeze(topk(pred, 1)[1].int()).cpu().detach().numpy() for pred in predictionList]

def binaryMetricsCompute(label: np.ndarray,
                         outIndex: np.ndarray,
                         outSoftmax: np.ndarray) -> None:
    print(f'''Accuracy: {accuracy_score(label, outIndex)};\nROC AUC score: {roc_auc_score(label, outSoftmax[:, 1])}, {roc_auc_score(label, outSoftmax[:, 0])};\nF1: {f1_score(label, outIndex)}''')
    print(f'''Precision: {precision_score(label, outIndex)};\nRecall: {recall_score(label, outIndex)}''')

def metricsIndicator(label: np.ndarray,
                     pred: np.ndarray,
                     softmax: np.ndarray) -> dict:
    return {"Accuracy": accuracy_score(label, pred),
            "ROC_AUC_score": roc_auc_score(label, softmax[:, 1]),
            "F1": f1_score(label, pred),
            "Precision": precision_score(label, pred),
            "Recall": recall_score(label, pred)}