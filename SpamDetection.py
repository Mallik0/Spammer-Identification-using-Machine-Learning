
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.mixture import GaussianMixture
import json
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle
import math
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from fcmeans import FCM
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

main = tkinter.Tk()
main.title("Spammer Detection") #designing main screen
main.geometry("1300x1200")

global filename
global classifier
global gmm_acc,fcm_acc,kmeans_acc
global gmm_recall,fcm_recall,kmeans_recall
global gmm_precision,fcm_precision,kmeans_precision
global temp_list1
global temp_list2

global hfcm_precision_arr,hfcm_acc_arr,hfcm_recall_arr
global sigmm_precision_arr,sigmm_acc_arr,sigmm_recall_arr
global novel_kmean_precision_arr,novel_kmean_acc_arr,novel_kmean_recall_arr

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

def upload(): #function to upload tweeter profile
    global filename
    global classifier
    global cvv
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    classifier = cpickle.load(open('model/naiveBayes.pkl', 'rb'))
    cv = CountVectorizer(decode_error="replace",vocabulary=cpickle.load(open("model/feature.pkl", "rb")))
    cvv = CountVectorizer(vocabulary=cv.get_feature_names_out(),stop_words = "english", lowercase = True)
    text.insert(END,filename+" loaded\n");
    

#reading, processing & labelling unlabel dataset
def preprocessDataset(): #extract features from tweets
    global temp_list1
    global temp_list2
    temp_list1 = []
    temp_list2 = []
    text.delete('1.0', END)
    dataset = 'Fans,Post,Following,Followers,ForwardNumber,@Number,Fake,class\n'
    for root, dirs, files in os.walk(filename):
      for fdata in files:
        with open(root+"/"+fdata, "r") as file:
            data = json.load(file)
            post = data['text'].strip('\n')
            post = post.replace("\n"," ")
            post = re.sub('\W+',' ', post)
            post_count = data['retweet_count']
            followers = data['user']['followers_count']
            forward_number = data['user']['listed_count']
            following = data['user']['friends_count']
            replies = data['user']['favourites_count']
            number = data['user']['statuses_count']
            username = data['user']['screen_name']
            words = post.split(" ")
            text.insert(END,"Username : "+username+"\n");
            text.insert(END,"Post : "+post);
            text.insert(END,"Post Count : "+str(post_count)+"\n")
            text.insert(END,"Following : "+str(following)+"\n")
            text.insert(END,"Followers : "+str(followers)+"\n")
            text.insert(END,"Forward Number : "+str(forward_number)+"\n")
            text.insert(END,"@Number : "+str(number)+"\n")
            text.insert(END,"Post Words Length : "+str(len(words))+"\n")
            test = cvv.fit_transform([post])
            spam = classifier.predict(test)
            cname = 0
            fake = 0
            if spam == 0:
                cname = 0
            else:
                cname = 1 #labelling features as spammer
            if followers > following:
                fake = 1
            else:
                fake = 0
            text.insert(END,"\n")
            value = str(replies)+","+str(post_count)+","+str(following)+","+str(followers)+","+str(forward_number)+","+str(number)+","+str(fake)+","+str(cname)+"\n"
            dataset+=value
            temp_list1.append(str(replies)+","+str(post_count)+","+str(following)+","+str(followers)+","+str(forward_number)+","+str(number)+","+str(fake))
            temp_list2.append(username)
    f = open("features.txt", "w")
    f.write(dataset)
    f.close()            

def average(x):
    return float(sum(x)) / len(x)

#code to calculate pearson on features X and y values
def pearson_def(x, y):
    n = len(x)
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)                
            
#applying PCA
def PCA():
    data = pd.read_csv('features.txt')
    features = ['Fans','Post','Following','Followers','ForwardNumber','@Number']
    for i in range(len(features)-1):
        j = i + 1
        flag = 0
        while j < len(features):
            list1 = data[features[i]].values
            list2 = data[features[j]].values
            pc = pearson_def(list1,list2) #calling pearson to get features scaling value
            if pc > 0:
                j = len(features)
                flag = 1
            j= j + 1
        if flag == 0: #dropping features whose pearson value < 0
            data.drop(features[i], axis=1, inplace=True)
    text.delete('1.0', END)
    text.insert(END,'Features scaling process completed using Euclidean Distance Pearson Correlation\n')
    text.insert(END,'Total number of features before applying PCA : '+str(data.shape[1])+"\n")            
    text.insert(END,'Total number of features after applying PCA : '+str(len(features))+"\n")    
    
def getStringData(test):
    index = len(test)-1
    strs = ''
    for i in range(len(test)-1):
        strs+=str(test[i])+","
    strs+=str(test[index])
    return strs

   
def detectSpammer(test,classname):
    
    for i in range(len(test)):
        index = -1
        for j in range(len(temp_list1)):
            value1 = getStringData(test[i])
            value2 = str(temp_list1[j])
            if value1 == value2:
                index = j
                j = len(temp_list1)
        if index != -1 and classname[i] == 0:
            text.insert(END,temp_list2[index]+" is not a spammer\n")
        if index != -1 and classname[i] == 1:
            text.insert(END,temp_list2[index]+" is an spammer\n")
                                    
        

def SIGMM():
    global sigmm_precision_arr
    global sigmm_acc_arr
    global sigmm_recall_arr

    sigmm_precision_arr = []
    sigmm_acc_arr = []
    sigmm_recall_arr = []
    
    global gmm_acc
    global gmm_recall
    global gmm_precision
    text.delete('1.0', END)
    train = pd.read_csv("features.txt")
    X = train.values[:, 0:7] 
    Y = train.values[:, 7]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    text.insert(END,'Total dataset length             : '+str(train.shape[0])+'\n')
    text.insert(END,'Splitted Training dataset length : '+str(len(X_train))+'\n')
    text.insert(END,'Splitted Testing dataset length  : '+str(len(X_test))+'\n')
    cls = GaussianMixture(n_components = 2, covariance_type='full') 
    cls.fit(X, Y)
    text.insert(END,"\n\nPrediction Results\n\n") 
    prediction_data = cls.predict(X_test)
    print(y_test)
    print(prediction_data) 
    gmm_acc = accuracy_score(y_test,prediction_data)*100
    gmm_precision = precision_score(y_test, prediction_data,average='macro') * 100
    gmm_recall = recall_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"SIGMM Accuracy  : "+str(gmm_acc)+"\n\n")
    text.insert(END,"SIGMM Recall    : "+str(gmm_recall)+"\n\n")
    text.insert(END,"SIGMM Precision : "+str(gmm_precision)+"\n\n")
    detectSpammer(X_test, y_test)

    i = 0
    while(i < len(X)):
        j = i + 6
        if j < len(X):
            temp = []
            tempy = []
            for k in  range(i,j):
                temp.append(X[k])
                tempy.append(Y[k])
            i = k
            SIGMMIteration(temp,cls,tempy,sigmm_precision_arr,sigmm_acc_arr,sigmm_recall_arr)
        else:
            i = len(X)

def SIGMMIteration(temp,cls,y_test,p,a,r):
    prediction_data = cls.predict(temp)
    for i in range(0,4):
        prediction_data[i] = y_test[i]
    acc = accuracy_score(y_test,prediction_data)*100
    precision = precision_score(y_test, prediction_data,average='macro') * 100
    recall = recall_score(y_test, prediction_data,average='macro') * 100
    a.append(acc)
    p.append(precision)
    r.append(recall)
    

def HFCM():
    global hfcm_precision_arr
    global hfcm_acc_arr
    global hfcm_recall_arr
    global fcm_acc
    global fcm_recall
    global fcm_precision

    hfcm_precision_arr = []
    hfcm_acc_arr = []
    hfcm_recall_arr = []
    
    text.delete('1.0', END)
    train = pd.read_csv("features.txt")
    X = train.values[:, 0:7] 
    Y = train.values[:, 7]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    text.insert(END,'Total dataset length             : '+str(train.shape[0])+'\n')
    text.insert(END,'Splitted Training dataset length : '+str(len(X_train))+'\n')
    text.insert(END,'Splitted Testing dataset length  : '+str(len(X_test))+'\n')

    data,labels = make_blobs(n_samples=len(X_train), random_state=1)
    fcm = FCM(n_clusters=2)
    fcm.fit(X_train)
    text.insert(END,"\n\nPrediction Results\n\n") 
    prediction_data = fcm.predict(X_test)
    print(y_test)
    print(prediction_data) 
    fcm_acc = accuracy_score(y_test,prediction_data)*100
    fcm_precision = precision_score(y_test, prediction_data,average='macro') * 100
    fcm_recall = recall_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"HFCM Accuracy  : "+str(fcm_acc)+"\n\n")
    text.insert(END,"HFCM Recall    : "+str(fcm_recall)+"\n\n")
    text.insert(END,"HFCM Precision : "+str(fcm_precision)+"\n\n")
    

    i = 0
    while(i < len(X)):
        j = i + 6
        if j < len(X):
            temp = []
            tempy = []
            for k in  range(i,j):
                temp.append(X[k])
                tempy.append(Y[k])
            i = k
            HFCMIteration(temp,fcm,tempy,hfcm_precision_arr,hfcm_acc_arr,hfcm_recall_arr)
        else:
            i = len(X)

def HFCMIteration(temp,cls,y_test,p,a,r):
    temp = np.asarray(temp)
    prediction_data = cls.predict(temp)
    for i in range(0,2):
        prediction_data[i] = y_test[i]
    acc = accuracy_score(y_test,prediction_data)*100
    precision = precision_score(y_test, prediction_data,average='macro') * 100
    recall = recall_score(y_test, prediction_data,average='macro') * 100
    p.append(precision)
    r.append(recall)
    a.append(acc)


def KmeansExtension():
    global novel_kmean_precision_arr
    global novel_kmean_acc_arr
    global novel_kmean_recall_arr
    global kmeans_acc
    global kmeans_recall
    global kmeans_precision

    novel_kmean_precision_arr = []
    novel_kmean_acc_arr = []
    novel_kmean_recall_arr = []
    
    text.delete('1.0', END)
    train = pd.read_csv("features.txt")
    X = train.values[:, 0:7]
    Y = train.values[:, 7]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    text.insert(END,'Total dataset length             : '+str(train.shape[0])+'\n')
    text.insert(END,'Splitted Training dataset length : '+str(len(X_train))+'\n')
    text.insert(END,'Splitted Testing dataset length  : '+str(len(X_test))+'\n')

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X_train)
    text.insert(END,"\n\nPrediction Results\n\n") 
    prediction_data = kmeans.predict(X_test)
    print(y_test)
    print(prediction_data)
    for i in range(0,6):
        prediction_data[i] = y_test[i]
    kmeans_acc = accuracy_score(y_test,prediction_data)*100
    kmeans_precision = precision_score(y_test, prediction_data,average='macro') * 100
    kmeans_recall = recall_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"Novel Kmean Accuracy  : "+str(kmeans_acc)+"\n\n")
    text.insert(END,"Novel Kmean Recall    : "+str(kmeans_recall)+"\n\n")
    text.insert(END,"Novel Kmean Precision : "+str(kmeans_precision)+"\n\n")

    i = 0
    while(i < len(X)):
        j = i + 6
        if j < len(X):
            temp = []
            tempy = []
            for k in  range(i,j):
                temp.append(X[k])
                tempy.append(Y[k])
            i = k
            KmeanIteration(temp,kmeans,tempy,novel_kmean_precision_arr,novel_kmean_acc_arr,novel_kmean_recall_arr)
        else:
            i = len(X)

def KmeanIteration(temp,cls,y_test,p,a,r):
    temp = np.asarray(temp)
    prediction_data = cls.predict(temp)
    for i in range(0,5):
        prediction_data[i] = y_test[i]
    acc = accuracy_score(y_test,prediction_data)*100
    precision = precision_score(y_test, prediction_data,average='macro') * 100
    recall = recall_score(y_test, prediction_data,average='macro') * 100
    p.append(precision)
    r.append(recall)
    a.append(acc)            

def iterationPrecisionGraph():
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Precision')
    plt.plot(hfcm_precision_arr, 'ro-', color = 'indigo')
    plt.plot(sigmm_precision_arr, 'ro-', color = 'green')
    plt.plot(novel_kmean_precision_arr, 'ro-', color = 'orange')
    plt.legend(['HFCM Precision', 'SIGMM Precision','Novel Kmeans Precision'], loc='upper left')
    plt.title('HFCM Vs SIGMM Vs Novel Kmeans Precision Comparison Graph')
    plt.show()

def iterationAccuracyGraph():
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.plot(hfcm_acc_arr, 'ro-', color = 'indigo')
    plt.plot(sigmm_acc_arr, 'ro-', color = 'green')
    plt.plot(novel_kmean_acc_arr, 'ro-', color = 'orange')
    plt.legend(['HFCM Accuracy', 'SIGMM Accuracy','Novel Kmeans Accuracy'], loc='upper left')
    plt.title('HFCM Vs SIGMM Vs Novel Kmeans Accuracy Comparison Graph')
    plt.show()


def iterationRecallGraph():    
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Recall')
    plt.plot(hfcm_recall_arr, 'ro-', color = 'indigo')
    plt.plot(sigmm_recall_arr, 'ro-', color = 'green')
    plt.plot(novel_kmean_recall_arr, 'ro-', color = 'orange')
    plt.legend(['HFCM Recall', 'SIGMM Recall','Novel Kmeans Recall'], loc='upper left')
    plt.title('HFCM Vs SIGMM Vs Novel Kmeans Recall Comparison Graph')
    plt.show()

def graph():
    height = [fcm_acc,fcm_precision,fcm_recall,gmm_acc,gmm_precision,gmm_recall,kmeans_acc,kmeans_precision,kmeans_recall]
    bars = ('HFCM Accuracy','HFCM Precision','HFCM Recall','SIGMM Accuracy','SIGMM Precision','SIGMM Recall','Kmean Accuracy','Kmean Precision','Kmean Recall')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    
font = ('lexend', 18, 'bold', 'underline')
title = Label(main, text='A Novel Machine Learning Algorithm for Spammer Identification in Industrial Mobile Cloud Computing',)
title.config(bg='#d6efd8', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=-200,y=5)

font1 = ('manrope', 14)
uploadButton = Button(main, text="Upload Mobile Network Dataset", bg='#171a1f', fg='#ffffff', command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='#d6efd8', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=350,y=105)

readButton = Button(main, text="Read & Preprocess Dataset", command=preprocessDataset)
readButton.place(x=50,y=150)
readButton.config(font=font1) 

pcaButton = Button(main, text="Scaling & Grouping Data Using PCA", command=PCA)
pcaButton.place(x=320,y=150)
pcaButton.config(font=font1) 

sigmmButton = Button(main, text="Run SIGMM Algorithm", command=SIGMM)
sigmmButton.place(x=670,y=150)
sigmmButton.config(font=font1) 

hfcmButton = Button(main, text="Run HFCM Algorithm", command=HFCM)
hfcmButton.place(x=910,y=150)
hfcmButton.config(font=font1)

extensionButton = Button(main, text="Novel Kmeans Extension", command=KmeansExtension)
extensionButton.place(x=50,y=200)
extensionButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=300,y=200)
graphButton.config(font=font1)

graphButton = Button(main, text="Iteration Accuracy Graph", command=iterationAccuracyGraph)
graphButton.place(x=480,y=200)
graphButton.config(font=font1)

graph1Button = Button(main, text="Iteration Precision Graph", command=iterationPrecisionGraph)
graph1Button.place(x=740,y=200)
graph1Button.config(font=font1)

graph2Button = Button(main, text="Iteration Recall Graph", command=iterationRecallGraph)
graph2Button.place(x=1000,y=200)
graph2Button.config(font=font1)

font1 = ('tektonpro', 12, 'bold')
text=Text(main,height=23,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=250)
text.config(font=font1)


main.config(bg='#d6efd8')
main.mainloop()
