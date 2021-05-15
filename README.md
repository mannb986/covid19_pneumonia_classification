# Covid-19 and Pneumonia Classification with Deep Learning

The task here is to develop a learning model to support doctors with diagnosing illnesses that affect patients' lungs. I will use a set of data provided that is X-rays of lung scans with examples of patients who had either pneumonia, COVID-19, or no illness.  

Using Keras module, I will create a classification model that outputs a diagnosis based on a patient's X-ray scan.

The original datasets contained:

* Train
  * Covid
  * Normal
  * Pneumonia

* Test
  * Covid
  * Normal
  * Pneumonia


I used a CNN with the following layers:

`model.add(InputLayer(input_shape=(212, 212, 3)))
model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))`

The model resulted in a accuracy of the validation data of over 87%. 
