
"""
Yoga_Text_to_Class_Predict

Utility NN Model to take text description as input and generate
yoga class name as output.
"""


import os
import pandas as pd
import tensorflow as tf
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Embedding,GRU
import numpy as np
import datetime
import pickle

def make_model():
    learning_rate = 0.001
    model = Sequential()
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(34,activation='softmax'))
    opt = tf.optimizers.RMSprop(learning_rate= learning_rate)

    model.build(input_shape=(1,405)) # 405 is the size of bow representation
    return model


def bow_transform(df=None):
    '''
    Args: df : train/dev/text dataframe
    Returns: Bag of Words representation of X.
    '''

    text = df['text']
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
    cv.fit(text)
    text_counts= cv.transform(text).toarray()
    vocab = cv.vocabulary_
    print(vocab)

    X = text_counts
    return X,cv

def run():
    tlist = []
    classes = []
    path=os.getcwd()+"/data/yoga/txt"
    files = os.listdir(path)
    print(files)
    for fname in files:
        if fname.endswith(".txt"):
            f=open(os.path.join(path,fname),"r")
            txt = f.read()
            for i in range(50):
                tlist.append(txt)
                classes.append(fname.split(".")[0])


    df=pd.DataFrame(list(zip(tlist,classes)),columns=['text','classes'])



    X,cv = bow_transform(df)

    print(X.shape)
    print(X[3])

    Y = pd.get_dummies(df['classes'],columns=df["classes"].unique())
    print(Y)


    print(X.shape,Y.shape)

    learning_rate = 0.001
    batch_size=100
    epochs = 10
    #model = Text_Model()

    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size= 0.2,shuffle=True)

    opt = tf.optimizers.RMSprop(learning_rate= learning_rate)
    model=make_model()
    model.compile(optimizer=opt,loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    y_train_tensor = tf.convert_to_tensor(y_train.to_numpy(), dtype=tf.float32)

    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    model.fit(X_train_tensor,y_train_tensor,batch_size=batch_size,epochs=epochs,validation_split=0.2)

    y_test_tensor = tf.convert_to_tensor(y_test.to_numpy(), dtype=tf.float32)

    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    model.evaluate(X_test_tensor,y_test_tensor)

    sample= 'Keep the heart open and the gaze should be slightly over the top shoulder. On the inhale, elongate the spine and on the exhale take the twist slightly deeper.From Revolved Squatting Toe Balance</a>, bring the top arm up and around the back to meet the bottom arm that will reach under and wrap around its respective knee to be met at the wrist by the top hand.'
    print(isinstance(sample,str))
    sample = cv.transform([sample]).toarray()
    print(sample.shape)
    y_predicted = model.predict(sample)




    model.summary()

    Y.columns[np.argmax(y_predicted)]


    dt_string = datetime.datetime.now().strftime("%d%m%Y_%H%M_%S")
    model.save(os.getcwd()+"/final_models/text_model_"+dt_string+".h5")


    #Using Functional instead of subclassed model and model save and load code added.

    pickle_path=os.getcwd()+"/final_models/cv_pickle.pk" # to save bow count vectorizer object

    with open(pickle_path,"wb") as f:
        pickle.dump(cv,f)





def load_text_model(yoga_text,pickle_path,yoga_class):
    model=make_model()
    with open(pickle_path,"rb") as f:
        cv=pickle.load(f)

    yoga_text = cv.transform([yoga_text]).toarray()
    model.load_weights(pickle_path)
    #os.getcwd()+"/final_models/text_model_18012020_1127_26.h5"
    # use existing text model pickle or else use your own  pickle_path above
    return yoga_class[np.argmax(model.predict(yoga_text))]
