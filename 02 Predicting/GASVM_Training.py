import pandas as pd
from sklearn.svm import SVC
from sko.GA import GA

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib


qinling = pd.read_csv('mag_radio_train.csv')
data = qinling.loc[:,["1","2","3","5","7","8","9","10"]]
target = qinling.iloc[:, 10]
print('Data head:\n', data.head())
print('Target head:\n', target.head())


groupby_data_o=qinling.groupby(["Labels"])["Labels"].count()
print(groupby_data_o)


smo =  SMOTE(sampling_strategy=1/6, random_state=42)
data_smo, target_smo = smo.fit_resample(data, target)
df_smote = pd.concat([data_smo, target_smo], axis=1)

groupby_data_o=df_smote.groupby(["Labels"])["Labels"].count()
print(groupby_data_o)


UnderSampler = RandomUnderSampler(sampling_strategy=1/2)
data_smo_under, target_smo_under = UnderSampler.fit_resample(data_smo, target_smo)

df_smote_under = pd.concat([data_smo_under, target_smo_under],axis=1)  # 将特征和标签重新拼接
groupby_data_o_under = df_smote_under.groupby(['Labels'])['Labels'].count()  # 查看标签类别个数
print(groupby_data_o_under)



data_train, data_test, target_train, target_test = \
    train_test_split(data_smo_under, target_smo_under, train_size=0.8, random_state=42)


def cm_plot(target_test, target_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(target_test, target_pred)
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[y, x], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Accuracy=%f,Recall=%f,F1=%f" \
              % (precision_score(target_test, target_pred), \
                 recall_score(target_test, target_pred), \
                 f1_score(target_test, target_pred)))
    return plt



def function(p):
    c,g = p
    svm_model = SVC(C=c, gamma=g, kernel="rbf",probability=True)
    svm_model.fit(data_train,target_train)
    target_pred = svm_model.predict(data_test)
    # print(target_pred)

    # print("accuracy=",precision_score(target_test, target_pred))
    # print("Recall-score=", recall_score(target_test, target_pred))
    # print("F1-score=", f1_score(target_test, target_pred))


    point_all=len(target_pred)
    point_1=np.sum(target_pred==1)
    factor=point_1/point_all
    if factor<0.2:
        factor=10.0
    # print(point_all,point_1,point_1/point_all)
    # print(factor*12.0-f1_score(target_test, target_pred))
    # print("*******************************")

    return factor*12.0-f1_score(target_test, target_pred)


ga = GA(func=function,n_dim=2,size_pop=20,max_iter=120,prob_mut=0.1,lb=[1e-7,1e-7],ub=[10.0,10.0],precision=1e-2)
best_x,best_y = ga.run()
print("best_x=",best_x,"\n","best_y=",best_y)


svm_model = SVC(C=best_x[0], gamma=best_x[1],kernel="rbf",probability=True)
svm_model.fit(data_train,target_train)
print("SAVE Model")
joblib.dump(svm_model,"GASVM_magentic_radioactivity.pkl")
#
target_pred = svm_model.predict(data_test)
print("Accuracy=",precision_score(target_test, target_pred))
print("Recall-score=", recall_score(target_test, target_pred))
print("F1-score=",f1_score(target_test, target_pred))
print("R2-score=", r2_score(target_test, target_pred))
fpr, tpr, thresholds = metrics.roc_curve(target_test, target_pred, pos_label=1)
auc = metrics.roc_auc_score(target_test, target_pred, average='macro')
print("AUC=", metrics.auc(fpr, tpr))
# cm_plot(target_test, target_pred).show()
cm_plot(target_test, target_pred).savefig('GASVM_magentic_radioactivity_CM.png')
