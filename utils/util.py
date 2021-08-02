import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve ###计算roc和auc

def plain_auc(x,y,auc,label_name,xlabel,ylabel,title,color,save):
    plt.cla()
    plt.plot(x, y, color = color ,linestyle='--', 
                 label = f'{label_name} (area = {auc:.2f}, {auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(save, dpi=120)
    plt.show()

def plain_evl_result(dataframe,target,name,at_n):
    #### cal auc curve ####
    df = dataframe.copy()
    df = df.sort_values(by='R2_score',ascending=False)
    df['score'] =  range(len(df))
    df = df.sort_values(by='score', ascending=False)
    df['bias'] = df['bias'].astype(int)
    
    ## Roc curve
    fpr,tpr,threshold = roc_curve(df['bias'], df['score']) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    print(f'Roc curve score: {roc_auc:.3f}')
    plain_auc(fpr,tpr,roc_auc,'Roc curve','False Positive Rate','True Positive Rate',
              title='Model of Roc curve',color='green',save=os.path.join(target,name,'Model_Roc_curve.png'))
    ## PR curve
    precision, recall, threshold = precision_recall_curve(df['bias'], df['score'])
    PR_auc = auc(recall,precision)
    print(f'PR curve score: {PR_auc:.3f}')
    plain_auc(recall,precision,PR_auc,'PR curve','Recall','Precision',
              title='Model of PR curve',color='darkorange',save=os.path.join(target,name,'Model_PR_curve.png'))
    
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