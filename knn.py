import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#KNN
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#Cargar datos

df = pd.read_csv('clicks.csv',header = 0)
del df['checkin']
del df['checkout']
#print(df.head())
#Pintar mapa de calor
#sns.set()
#sns.heatmap(df.corr(), square=True, annot=True)
#plt.show()

#------Dividir conjunto de datos para train y test

train, test = train_test_split(df[['hotel_id','days','children', 'sale']], test_size=0.1)
train.reset_index(inplace = True)
test.reset_index(inplace = True)

#------KNN basado en vecinos

cv = KFold(n_splits = 5, shuffle = True)

for i, weights in enumerate(['uniform', 'distance']):
   total_scores = []
   for n_neighbors in range(1,30):
       fold_accuracy = []
       knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
       for train_fold, test_fold in cv.split(train):
          # division train test aleatoria
          f_train = train.loc[train_fold]
          f_test = train.loc[test_fold]
          # entrenamiento y ejecucion del modelo
          knn.fit( X = f_train.drop(['sale'], axis=1), 
                               y = f_train['sale'])
          y_pred = knn.predict(X = f_test.drop(['sale'], axis = 1))
          # evaluacion del modelo
          acc = accuracy_score(f_test['sale'], y_pred)
          fold_accuracy.append(acc)
       total_scores.append(sum(fold_accuracy)/len(fold_accuracy))
   
   #plt.plot(range(1,len(total_scores)+1), total_scores, marker='o', label=weights)
   print ('Max Value ' +  weights + " : " +  str(max(total_scores)) +" (" + str(np.argmax(total_scores) + 1) + ")")
   #plt.ylabel('Acc')      

#plt.legend()
#plt.show()
#----------------Construccion y ejecucion del modelo

# constructor
n_neighbors = 30
weights = 'uniform'
knn = neighbors.KNeighborsClassifier(n_neighbors= n_neighbors, weights=weights) 
# fit and predict
knn.fit(X = train[['hotel_id','days','children']], y = train['sale'])
y_pred = knn.predict(X = test[['hotel_id','days','children']])
acc = accuracy_score(test['sale'], y_pred)
print ('Acc', acc)