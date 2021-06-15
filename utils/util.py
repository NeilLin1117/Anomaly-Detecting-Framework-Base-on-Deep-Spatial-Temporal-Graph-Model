import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

def plain_evl_result(dataframe,target,name,at_n):
    #### cal auc curve ####
    df = dataframe.copy()
    df = df.sort_values(by='R2_score',ascending=False)
    df['score'] =  range(len(df))
    df = df.sort_values(by='score', ascending=False)
    df['bias'] = df['bias'].astype(int)
    fpr,tpr,threshold = roc_curve(df['bias'], df['score']) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    plt.plot(fpr, tpr, color = 'green' ,linestyle='--', 
                 label = 'Roc curve (area = %0.2f, ' % roc_auc + '%0.3f) ' % roc_auc)
    print(f'Roc curve score: {roc_auc:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model of Roc curve')
    plt.legend(loc="best")
    #plt.savefig('result/Model_Roc_by_hour.png', dpi=120)
    plt.savefig(os.path.join(target,name,'Model_Roc_curve.png'), dpi=120)
    plt.show()
    plt.cla()
    
    precision = []
    recall = []
    #### Precision@n and Recall@n ####
    relevant_item = (df['bias'] == 1).sum()
    for i in range(10,at_n+1,10):
        first_data = df[:i]
        recall_at = (first_data['bias'] == 1).sum() / relevant_item
        recall.append(round(recall_at,3))
        precision_at = (first_data['bias'] == 1).sum() / len(first_data)
        precision.append(round(precision_at,3))
        print(f'precision@{i}: {precision_at:.3f}')
        print(f'recall@{i}: {recall_at:.3f}')
    plt.plot(range(10,at_n+1,10) ,precision , color = 'blue' ,linestyle='-', label='precision@n')
    plt.scatter(range(10,at_n+1,10) , precision , c = 'blue',marker = '.')
    plt.plot( range(10,at_n+1,10), recall, color = 'red' ,linestyle='-', label='recall@n')
    plt.scatter(range(10,at_n+1,10) , recall, c = 'red',marker = '.')
    
    for k,i in enumerate(range(10,at_n+1,10)):
        plt.annotate(str(precision[k]), xy = (i, precision[k]), xytext = (i+0.5, precision[k]+0.01))
        plt.annotate(str(recall[k]), xy = (i, recall[k]), xytext = (i+0.5, recall[k]+0.01))
    plt.legend(loc="best")
    plt.xlabel('@n')
    plt.ylabel('rate')
    plt.title('precision@n and recall@n of device')
    plt.savefig(os.path.join(target,name,'precision@n_and_recall@n_of_device.png'), dpi=120)
    plt.show()