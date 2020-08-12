import numpy as np
from math import sqrt
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from itertools import cycle
from scipy import interp
from sklearn.preprocessing import label_binarize

Y_pred = np.load('0.6594_780pred.npy')
Y_valid = np.load('0.6594_780valid.npy')
# Y_pred = np.load('0.7124_1540pred.npy')
# Y_valid = np.load('0.7124_1540valid.npy')
print(Y_pred)
print(Y_valid)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i],Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(_)
fpr["micro"], tpr["micro"], _a = roc_curve(Y_valid.ravel(), Y_pred.ravel())
print(_a)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(5):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 5
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label=' ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'pink', 'red'])
for i, color in zip(range(5), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of Rank {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("ROC_7.18_4.pdf")
print("done")
