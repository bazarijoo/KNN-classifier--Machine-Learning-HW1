import LoadData
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn import metrics

def knnClassify(X_train, test_element, y_train,k):

    distances = []
    for train_element  in X_train:
        mul=(test_element - train_element) ** 2
        sum=mul.sum(axis=0)
        distances.append( round(sum**0.5,3) )

    distances = np.c_[distances, y_train].tolist()
    distances.sort(key=lambda x:x[0])

    # find the prediction according to first k's
    benign_freq=0
    malignant_freq=0

    for i in range(k):
        if distances[i][1]==2 :
            benign_freq+=1
        else:
            malignant_freq+=1

    if benign_freq > malignant_freq:
        prediction=2
    else:
        prediction=4

    return prediction


def predictClass(X_train, X_test, y_train,k):

    y_pred=[]
    # knnClassify(X_train, X_test[0], y_train, k)

    for test_element in X_test:
        y_pred.append(knnClassify(X_train, test_element, y_train,k))
    return y_pred

def calculateConfusionMetrics(y_pred,y_test):
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_test[i]==4:
            TP+=1
        elif y_pred[i] == y_test[i] == 2:
            TN+=1
        elif y_pred[i] ==4 and y_test[i]==2:
            FP+=1
        elif y_pred[i] == 2 and y_test[i]== 4:
            FN+=1

    accuracy=(TP+TN)/(TP+TN+FP+FN)
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    faults=FN+FP

    return accuracy,recall,precision,faults

def findPrediction(X_train, X_test, y_train):
    k=3
    K_values=[]
    faults=[]
    best_k=0
    y_pred=[]
    fault=1000

    while k<=15:
        new_y_pred = predictClass(X_train, X_test, y_train,k)
        _, _, _, new_fault = calculateConfusionMetrics(new_y_pred,y_test)
        faults.append(new_fault)
        K_values.append(k)

        if new_fault<fault:
            fault=new_fault
            y_pred=new_y_pred
            best_k=k

        k=k+2

    print("Best K for prediction is : "+ str(best_k))

    plt.title('Errors based on k')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.scatter(K_values, faults)
    plt.show()

    return y_pred

def PCAAnalysis(X,y,data):


    X = StandardScaler().fit_transform(X)
    pca = PCA(.80)
    X = pca.fit_transform(X)
    print("Number of features reduced to : "+str(X.shape[1]))

    principalDf = pd.DataFrame(data = X
                               , columns=['principal component 1', 'principal component 2','principal component 3'])

    finalDf = pd.concat([principalDf, data['Class']], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component PCA', fontsize=20)
    Classes = [2, 4]
    labels=['benign','malignant']
    colors = ['g', 'r']
    plt.title('PCA Analysis')

    for Class, color in zip(Classes, colors):
        indicesToKeep = finalDf['Class'] == Class
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , finalDf.loc[indicesToKeep, 'principal component 3']
                   , c=color
                   , s=50)
    ax.legend(labels)
    ax.grid()
    plt.show()

    return X

if __name__ == '__main__' :

    data=LoadData.loadDataset()

    X = data.iloc[:,:-1].values   # Here first : means fetch all rows :-1 means except last column
    y = data.iloc[:,10].values    # : is fetch all rows 10 means 10th column
    X = X[:, 1:]  # ignore ids

    PCAFeatures=PCAAnalysis(X, y, data)

    X_train, X_test, y_train, y_test =  train_test_split(PCAFeatures,y,test_size = 0.20, random_state = 5)

    y_pred=findPrediction(X_train, X_test, y_train)

    accuracy,recall,precision,_=calculateConfusionMetrics(y_pred,y_test)

    print('Manual Accuracy: {:.3f}'.format(accuracy))
    print('Manual Recall: {:.3f}'.format(recall))
    print('Manual Precison: {:.3f}'.format(precision))


    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.transform(X_test)
    #
    # classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    # classifier = classifier.fit(X_train, y_train)
    #
    # y_pred = classifier.predict(X_test)

    # check accuracy

    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # precision = metrics.precision_score(y_test, y_pred)
    # recall = metrics.recall_score(y_test, y_pred)

    # print()
    # print('Sklearn Accuracy: {:.3f}'.format(accuracy))
    # # print('Sklearn Recall: {:.3f}'.format(recall))
    # print('Sklearn Precison: {:.2f}'.format(precision))

