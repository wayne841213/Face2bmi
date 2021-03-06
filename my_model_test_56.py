# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:35:12 2020

@author: NTUME138
"""

import pandas as pd
import numpy as np 

dataset = pd.read_csv('./normalized_data.csv')

y = dataset['bmi']

for n,data in dataset.iterrows():
    bmi = np.round(float(data['bmi']))
    if bmi<15:
        dataset['bmi'][n]=15
    elif 70<bmi:
        dataset['bmi'][n]=70

dataset['bmi'] = np.round(dataset['bmi'].astype(float)).astype('str')

Label=[15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27.,
       28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40.,
       41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53.,
       54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65., 66.,
       67., 68., 69., 70.]

classes=np.array(Label).astype('str').tolist()

#%%

from keras.models import load_model


model = load_model('my_model_56_f.h5')


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

#%% this is the augmentation configuration we will use for training
from keras.preprocessing.image import ImageDataGenerator


test_datagen = ImageDataGenerator(
        rescale=1./255)


validation_generator = test_datagen.flow_from_dataframe(
        dataframe=dataset,
        directory='normalized_images',
        x_col="name",
        y_col="bmi",
        target_size=(299, 299),
        batch_size=1,
        classes=classes,
        shuffle=False)


nb_validation_samples = len(validation_generator)  






prob = model.predict_generator(validation_generator,steps=nb_validation_samples
                               ,verbose=1)

predict_label = np.argmax(prob, axis=1)


true_label = validation_generator.classes


loss, acc = model.evaluate_generator(validation_generator,
                                     steps=nb_validation_samples,
                                     verbose=1
                                     )

print('Loss : %f  Accuracy : %f' %(loss,acc))

#%%

predict=[]

for n in predict_label:
    try:
        predict.append(Label[n])
    except:
        1

predict = np.array(predict).astype(float)


tru=0
for pr,gt in zip(predict,y):
    l=abs(gt-pr)
    if l/gt < 0.1:
        tru+=1
        
true_rate = tru/nb_validation_samples

if len(predict)!=len(np.array(y)):
    print('Error: data lengths mismatch.')
else:
    print('true_rate : %f' %true_rate)