#!/usr/bin/env python
# coding: utf-8

# ## Kütüphaneleri yükleyelim
# 

# 
# #Data analiz kütüphaneleri
# import sys
# import pandas as pd
# import numpy as np
# import sklearn
# import matplotlib
# import keras

# In[22]:


#Görselleştirme kütüphaneleri
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import pandas as pd
import numpy as np


# In[10]:


#Datayı yükliyelim
kalp_hastalığı = pd.read_csv(r"C:\Users\cagla\Desktop\dosya\heart.csv")
kalp_hastalığı.head()


# In[11]:


#Datamızda nan value var mı bakalım
print(kalp_hastalığı.isnull().sum().sum())


# In[12]:


# Datayı şimdiden çevirelim
kalp_hastalığı = kalp_hastalığı.apply(pd.to_numeric)


# In[13]:


kalp_hastalığı.describe(include="all")


# In[14]:


#Datayı görselleştirmeden önce türkçeye çevirelim
#kalp_hastalığı = kalp_hastalığı.rename(columns={'age': 'yaş'}) bu şekilde de olur ama daha pratiği var
kalp_hastalığı.columns = ['yaş', 'cinsiyet', 'ağrı_seviyesi', 'kan_basıncı', 'kolestrol', 'kandaki_şeker', 'Elektrokardiyografi' , 'Max_Nabız', 'Anjina', 'St_depresyonu', 'Eğim', 'ca', 'Kalıtsal_kan_bozukluğu','Hedef']


# In[15]:


#ca'nın ne olduğunu tam olarak anlamadığımdan çevirmedim :)#
kalp_hastalığı.head()


# In[16]:


kalp_hastalığı.hist(figsize=(10,10))


# In[17]:


#Cinsiyete göre kolestrol dağılımı
sns.barplot(x="cinsiyet", y="kolestrol", data=kalp_hastalığı)


# In[18]:


#yaşa bağlı kolestrol grafiği
pd.crosstab(kalp_hastalığı.yaş,kalp_hastalığı.ağrı_seviyesi).plot(kind="bar",figsize=(15,10))
plt.title('Yaşa bağlı kolestrol grafiği')
plt.xlabel('Age')
plt.ylabel('Sıklık')
plt.show()


# In[19]:


#yaşa bağlı hastalık grafiği
pd.crosstab(kalp_hastalığı.yaş,kalp_hastalığı.Hedef).plot(kind="bar",figsize=(20,6))
plt.title('Yaşa bağlı hastalık grafiği')
plt.xlabel('Age')
plt.ylabel('Sıklık')
plt.show()


# In[20]:


plt.figure(figsize=(10,10))
sns.heatmap(kalp_hastalığı.corr(),annot=True,fmt='.1f')
plt.show()
# https://explained.ai/matrix-calculus/index.html bu makaleyi okuyarak alttaki tablo hakkında bilgi edinebilirsiniz 


# # Test ve Train data setlerini oluşturalım

# In[23]:


#hedefi kenidimiz tahmin ediceğimiz için hedef kolonunu kaldıralım
X = np.array(kalp_hastalığı.drop(['Hedef'], 1))
y = np.array(kalp_hastalığı['Hedef'])


# In[24]:


X


# In[25]:


y


# In[26]:


ortalama = X.mean(axis=0)
X -= ortalama
standart_sapma = X.std(axis=0)
X /= standart_sapma


# In[27]:


from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)


# In[28]:


# convert the data to categorical labels
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])


# In[29]:


X_train[0]


# In[30]:


#azıcık test ediyim dedim :)#
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_logreg)


# In[31]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_svc)


# ## Yapay sinir ağı oluşturarak doğruluk oranını arttırmaya çalışalım

# In[35]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
#Bende sadece keras yazınca AttributeError: module 'tensorflow.python.framework.ops' has no attribute '_TENSOR_LIKE_TYPES' bu şekilde bir hata verdi
#O yüzden tf.keres.models yazdım


# Model için adam algoritmasını kullandım
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    
    # compile model
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model = create_model()

print(model.summary())


# In[36]:


# fit the model to the training data
history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=50, batch_size=10)


# Yukarıda yaptığımız Lojistik Regresyon ve SVM algoritmalarından daha yüksek doğruluk oranına eriştik.

# In[66]:


# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[37]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[38]:


print(X_train)


# In[41]:


print(history.history['accuracy'])


# In[ ]:




