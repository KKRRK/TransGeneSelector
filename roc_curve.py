from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np


def calculate_tpr_fpr(label, scores, thresholds):
    tpr = []
    fpr = []

    for threshold in thresholds:
        TP = np.sum((scores >= threshold) & (label == 1))
        FP = np.sum((scores >= threshold) & (label == 0))
        FN = np.sum((scores < threshold) & (label == 1))
        TN = np.sum((scores < threshold) & (label == 0))
        tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))

    return fpr, tpr

thresholds = np.linspace(0, 1, 1000)

data = pd.read_csv('output_result.csv', index_col=None, header=0)

label = data['labels']
y_scores = [data.iloc[:, i] for i in range(1, data.shape[1])]
print(y_scores)

model_type = data.columns[1:]

#set color
colors = [
    'darkorange', 'green', 'red', 'blue', 'yellow', 'black', 'pink', 
    'purple', 'brown', 'gray', 'cyan', 'magenta', 'lime', 'navy', 
    'gold', 'silver', 'teal', 'indigo', 'violet', 'olive', 'peachpuff', 
    'chartreuse', 'fuchsia', 'darkgreen', 'lightblue', 'darkviolet', 
    'mediumspringgreen', 'deepskyblue', 'crimson', 'salmon', 'darkred', 
    'lightsalmon', 'lavender', 'lightgreen', 'mediumorchid', 'royalblue', 
    'rosybrown', 'firebrick', 'chocolate', 'mediumvioletred', 'goldenrod', 
    'cadetblue', 'seagreen', 'midnightblue', 'orchid', 'sienna', 
    'tomato', 'lightpink', 'palevioletred', 'mediumturquoise', 'steelblue', 
    'forestgreen', 'coral', 'sandybrown', 'mediumseagreen'
]
# print(y_scores)

plt.figure()

acc = []
pre = []
rec = []
f1 = []
auc_ = []
for i,scores in enumerate(y_scores):
    
    #caculate acc, pre, rec, f1
    y_pred = [1 if x >= 0.5 else 0 for x in scores]
    acc.append(accuracy_score(label, y_pred))
    pre.append(precision_score(label, y_pred))
    rec.append(recall_score(label, y_pred))
    f1.append(f1_score(label, y_pred))
    
    fpr, tpr = calculate_tpr_fpr(label, scores, thresholds)

    roc_auc = auc(fpr, tpr)
    auc_.append(roc_auc)
    

    plt.plot(fpr, tpr, color=colors[i], lw=2, label=model_type[i]+'(area = %f)' % roc_auc)
    
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc="lower right")

#save data
data = pd.DataFrame([acc, pre, rec, f1 ,auc_], index=['acc', 'pre', 'rec', 'f1', 'auc'], columns=model_type)
data.to_csv('output_five_index.csv', index=True)

plt.show()