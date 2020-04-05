# -*- coding: utf-8 -*-
"""

"""

# -*- coding: utf-8 -*-
"""

"""

import requests
import os
from sklearn.datasets import make_blobs
import gensim
from keras.utils.np_utils import to_categorical
from yellowbrick.cluster import KElbowVisualizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pandas as pd
from xml.dom import minidom
import xml.etree.ElementTree as ET
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from sklearn.pipeline import Pipeline  
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import numpy as np
from sklearn import model_selection
df_cols = ["headline","text", "bip:topics", "dc.date.published","itemid","XMLfilename"]
rows = []
path='C:\\Users\\Owner\\Documents\\Machine Learning\\Data2'
column_headline = np.array([])
column_itemid = np.array([])
column_text = np.array([])
column_bip_topics = np.array([])
column_dc_date_published = np.array([])
column_filename = np.array([])
files = []
array=[]
def ensembleclassification(x,y,Text,cluster):
    
    kfold_vc = model_selection.KFold(n_splits=2, random_state=10)
    estimators = []
    y=y.values.ravel()
    mod_lr = LogisticRegression()
    estimators.append(('logistic', mod_lr))
    mod_dt = DecisionTreeClassifier()
    estimators.append(('cart', mod_dt))
    mod_sv = SVC()
    estimators.append(('svm', mod_sv))
    # Lines 9 to 11
    ensemble = VotingClassifier(estimators)
    results_vc = model_selection.cross_val_score(ensemble, x, y, cv=kfold_vc)
    print("Accuracy for the cluster: %.6f%%." %cluster)
    print(results_vc.mean())
    
def classification(x,y,Text,cluster):   
    y=y.values.ravel()
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(x, y,random_state=1)
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
     # Train the data on a classifier #Naive Bayes
    classifier = Pipeline([
    ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None)))])
    feature_vector_train=train_x
    is_neural_net=False
    label=train_y
    feature_vector_valid=valid_x
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    accuracy = metrics.accuracy_score(predictions, valid_y)
    f1_score =metrics.f1_score(predictions, valid_y,average="macro")
    precision_score =metrics.precision_score(predictions, valid_y,average="macro")
    recall_score=metrics.recall_score(predictions, valid_y,average = "macro")
    print("Training on the classifier: ", accuracy)
    print("Training on the classifier: ", f1_score)
    print("Training on the classifier: ", precision_score)
    print("Training on the classifier: ", recall_score)
    

        
   
    
   
def create_cnn(X,Y):
    # Add an Input Layer
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
    from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
    from keras.callbacks import EarlyStopping
    from keras.models import Model,load_model
    epochs = 10
    emb_dim = 128
    batch_size = 256
    model = Sequential()
    model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.7))
    model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
    model.add(Dense(Y.shape[1], activation='softmax',name ='feature_dense'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    featuresnn = model.fit(X,Y, epochs=epochs, batch_size=batch_size,validation_split=0.4,callbacks=[EarlyStopping(monitor='val_loss',patience=3, min_delta=0.0001)])
    #intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('feature_dense').output)
    #intermediate_layer_model.summary()
    feauture_engg_data = model.predict(X)
    feauture_engg_data = pd.DataFrame(feauture_engg_data)
    print('feauture_engg_data shape:', feauture_engg_data.shape)
    print('New Features',feauture_engg_data)
    print('New Features',feauture_engg_data)
    import matplotlib.pyplot as plt


    
    max_features = 20000
    maxlen = 100
    batch_size = 32

    X_train, X_test, y_train, y_test = train_test_split(feauture_engg_data,Y, test_size=0.3, random_state=666)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    #Feature Extraction
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=Y.shape[1]))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(Y.shape[1], activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='sgd',metrics=['acc'])

    print('Train...')
    history=model.fit(X_train, y_train,batch_size=batch_size,epochs=10,validation_data=[X_test, y_test])
    accr = model.evaluate(X_test,y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()



 
def create_rnn_lstm(x,y,Text,cluster):  
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model


       

    
    
for r, d, f in os.walk(path):
    for file in f:
        if '.xml' in file:
            files.append(os.path.join(r, file))
for Single_file in files:
    tree = ET.parse(Single_file)
    root = tree.getroot()
    s_headline = root.find("headline").text
    column_headline = np.append(column_headline, s_headline)
    s_itemid = root.attrib.get("itemid")
    column_itemid = np.append(column_itemid, s_itemid)
    s_text = ""
    s_bip_topics = ""
    code3='0'
    bip_c=''
    for node in root:
        #print("tags: ",node.tag,"attribs: ",node.attrib)
        if(node.tag == 'text'):
            for textnode in node:
                if(textnode.tag == 'p'):
                    #print(dir(textnode))
                    s_text = s_text + " " + textnode.text
        elif(node.tag == 'metadata'):
            for metanode in node:
                #print(metanode.tag)
                if(metanode.tag == 'dc' and metanode.attrib.get("element") == 'dc.date.published'):
                    s_dc_date_published = metanode.attrib.get("value")
                elif(metanode.tag == 'codes' and metanode.attrib.get("class") == 'bip:topics:1.0'):
                    for bipnodes in metanode:  
                        #print(s_bip_topics)
                        s_bip_topics = bipnodes.attrib.get("code") + "," + s_bip_topics
                        if(code3=='0'):
                            
                            bip_c=bipnodes.attrib.get("code")
                            #print(bip_c)

    array.append(bip_c)
    #print(array)                        
    column_text = np.append(column_text, s_text)
    column_dc_date_published = np.append(column_dc_date_published, s_dc_date_published)
    column_bip_topics = np.append(column_bip_topics, s_bip_topics)
    s_xmlfilename = s_itemid + "newsML.xml"
    column_filename = np.append(column_filename, s_xmlfilename)
#print(column_headline)
#print(column_itemid)
#print(column_text.shape)
#print(column_bip_topics)
#print(column_dc_date_published)
#print(column_filename)
#Yfinal = bip()
Final_Table = np.column_stack([column_headline,column_text,column_bip_topics,column_dc_date_published,column_itemid,column_filename])
#print(Final_Table.shape)
Final_DF = pd.DataFrame(Final_Table, columns = df_cols)
Final_bp = pd.DataFrame(array, columns =[ 'bip_topics'])
#print(Final_DF)	
#print(Final_bp)


def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 = str1 + ele + " "   
    
    # return string   
    return str1  
#removing numbers

   
def stopwords():
    import nltk
    from nltk.corpus import stopwords 
    #nltk.download('stopwords')
    slist = stopwords.words('english')
    dfcols1=['itemid','text']
    
    rows1 = []
    
    for n in files:
        clean=[]
        treec = ET.parse(n)
        rt=treec.getroot()
        id = rt.get('itemid')
        for ch in rt:
            for neighbour in ch.iter('text'):
                for a in neighbour:
                    k=a.text
                    for word in k.split():
                        if word not in slist:
                            clean.append(word)
                            stext = clean
        rows1.append({"itemid":id,"text":stext})
        
    df = pd.DataFrame(rows1,columns= dfcols1)
    #print(df)
    return df

df=stopwords()

#converting the list to string

column_text1 = np.array([])
dfcols2 =['filtered_text']
i=1
for i in df.text:    
        original = listToString(i)  
        column_text1 =np.append(original,column_text1)

fdata = pd.DataFrame(column_text1,columns = dfcols2)
#print(fdata)

#removing numbers

cleandig = np.array([])
coldef = ['nonum_text']    
for p in fdata.filtered_text:
    fil = ''.join(c for c in p if not c.isdigit())
    cleandig = np.append(fil, cleandig)
numno = pd.DataFrame(cleandig,columns = coldef )
#print(numno)

#Applying stemming
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
porter = PorterStemmer()
lancaster=LancasterStemmer()
column_stem = np.array([])
dfcols2 =['stemmed_text']
for i in numno.nonum_text:
    stem = porter.stem(i)
    column_stem = np.append(stem,column_stem)
fdata2 = pd.DataFrame(column_stem,columns = dfcols2)
#print(fdata2)

#removing special charactera
from string import punctuation
coldef2 =['nopunc_text']
column_nopunc = np.array([])
from nltk.stem import PorterStemmer, WordNetLemmatizer

for pc in fdata2.stemmed_text:
    nopunc = ''.join(c for c in pc if c not in punctuation)
    sent_tokenized = nopunc.split(" ")
    lemmatizer = WordNetLemmatizer()
    no_punc = [lemmatizer.lemmatize(word) for word in sent_tokenized]
    column_nopunc = np.append(nopunc,column_nopunc)

fdata3 = pd.DataFrame(column_nopunc, columns = coldef2)
#print(fdata3)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tf= TfidfVectorizer(analyzer = 'word', ngram_range =(1,2), lowercase = True, max_features = 2000, min_df = 1)       
tf_transformer = tf.fit_transform(fdata3['nopunc_text'])
feature_tf=tf.get_feature_names()
features_df1=pd.DataFrame(tf_transformer.toarray(),columns =feature_tf )
features_df = pd.DataFrame(tf_transformer.toarray(),columns =feature_tf )


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification     
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_df, Final_bp, test_size=0.3)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn import cluster
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

mms = MinMaxScaler()
mms.fit(features_df)
data_transformed = mms.transform(features_df)
k=1
K=0
Sum_of_squared_distances = []
elbow= range(3,20)
for k in elbow:
    km = KMeans(k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
x=elbow
y=Sum_of_squared_distances
plt.plot(elbow, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
from kneed import KneeLocator
kn = KneeLocator(x, y, curve='convex', direction='decreasing',S=50)
kn.plot_knee_normalized()
kn.plot_knee()
print(kn.knee)
plt.show()


from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
X, y = make_blobs(n_samples=1000, n_features=12, centers=8, random_state=42)


# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(3,20), metric='distortion', timings=False, locate_elbow=True)

visualizer.fit(features_df)        # Fit the data to the visualizer
visualizer.show() 
#print(visualizer.elbow_value_)    
Optimal_NumberOf_Components=visualizer.elbow_value_
   # Finalize and render the figure
#Optimal_NumberOf_Components=clusters[Sum_of_squared_distances.index(min(Sum_of_squared_distances))]
#print(Optimal_NumberOf_Components)

#x = range(1, len(Sum_of_squared_distances)+1)
#y=Sum_of_squared_distances




silhouette_score_values=list()
    #from kmeansplots import kmeans_plot, silhouette_plot
clusters=range(3, 20)
    #num_clusters =2
for num_clusters in clusters:
    classifier=cluster.KMeans(num_clusters,init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True)
    n_clusters=num_clusters
    km = KMeans(n_clusters=num_clusters)
    #X=df
    y=km.fit(features_df)
    #labels= classifier.predict(df)
    labels = km.labels_
    centers = km.cluster_centers_
        # Create a dataframe for cluster_centers (centroids)
    score = silhouette_score (features_df, labels, metric='euclidean')
    silhouette_score_values.append(sklearn.metrics.silhouette_score(features_df,labels ,metric='euclidean', sample_size=None, random_state=None))
 
plt.xlabel('number of clusters k')
plt.ylabel('Silhouette score values')    
plt.plot(clusters, silhouette_score_values)
plt.title("Silhouette score values vs Numbers of Clusters ")
plt.show()
 
#Optimal_NumberOf_Components=clusters[silhouette_score_values.index(max(silhouette_score_values))]
#Optimal_NumberOf_Components=clusters[Sum_of_squared_distances.index(min(Sum_of_squared_distances))]

print ("Optimal number of components is:")
print ( Optimal_NumberOf_Components)


clusters = Optimal_NumberOf_Components
cluster_ids = np.array([])  
kmeans = KMeans(n_clusters = clusters) 
clustering = kmeans.fit(features_df) 
clusters = kmeans.labels_.tolist()
centroids =  kmeans.cluster_centers_
#print(clusters)
centroids = kmeans.cluster_centers_



from sklearn.utils.extmath import randomized_svd

U, Sigma, VT = randomized_svd(tf_transformer, n_components=3, n_iter=100,
 random_state=122)

#printing the concepts


#Topics Visualization

import umap
import matplotlib as mpl
import matplotlib.pyplot as plt
X_topics=U*Sigma
embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=12).fit_transform(X_topics)




#Arranging clusters to a dataframe
cl_cols = ["cluster_id"]
for cl in clusters:
    cluster_ids = np.append(cl, cluster_ids)

cluster_frame = pd.DataFrame(cluster_ids,columns = cl_cols)
#print(cluster_frame)

plt.figure()
plt.figure(figsize=(7,5))
cmap = plt.cm.get_cmap('jet', cluster_frame.nunique())
plt.scatter(embedding[:, 0], embedding[:, 1], c = clusters,cmap='rainbow', s = 50, edgecolor='none')
print(embedding)
#plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1],c = clusters,cmap='rainbow',  marker = "x", s=150, linewidths = 5, zorder = 10)
plt.show()









cluster_table_cols = ["cluster_ids","text","biptopics"]

cluster_table = np.column_stack([cluster_ids,fdata3,array])
#print(cluster_table)
features_df['cluster_ids']=cluster_frame
features_df['Bip_topics']=Final_bp
features_df['Text']=fdata3['nopunc_text']
#print(features_df)


#ensembleclassification(x,y)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
clus=0
n=len(features_df.columns) 
x=pd.DataFrame()
while(clus<Optimal_NumberOf_Components):
    x=pd.DataFrame(features_df[features_df['cluster_ids'] == clus])
    datax=x[x.columns[0:n-3]]
    datay=pd.DataFrame(x[x.columns[n-2:n-1]])
    Text=pd.DataFrame(x[x.columns[n-1:n]])

    n_most_common_words = 8000
    max_len = 130
    tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True)
    tokenizer.fit_on_texts(Text.values.tolist())
    sequences = tokenizer.texts_to_sequences(Text.values.tolist())
    word_index = tokenizer.word_index
    #print('Found %s unique tokens.' % len(word_index))
    X = pad_sequences(sequences, maxlen=max_len)
    #print(X)
    code = np.array(datay)
    label_encoder = LabelEncoder()
    vec = label_encoder.fit_transform(datay)
    X = np.reshape(datax, (datax.shape[0], datax.shape[1]))
    Y = to_categorical(vec)
    ensembleclassification(datax,datay,Text,clus)
    
    create_cnn(X,Y)
    clus=clus+1


                                