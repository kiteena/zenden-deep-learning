import psycopg2
import numpy as np
from sklearn.utils import shuffle

def makeDatabaseConnection(connection_file): 
    file = open(connection_file, 'r')
    database = file.readline().strip()
    user = file.readline().strip()
    password = file.readline().strip()
    host = file.readline().strip()
    port = file.readline().strip()
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    return conn.cursor()

def fetchData(curr): 
    curr.execute('SELECT url_path,label FROM images where url_path is not null and label!=11')
    image_urls = curr.fetchall()
    images = shuffle(image_urls, random_state=40)
    data = [list(d) for d in zip(*images)]
    return data[0], data[1]

def enumerateSampleDistribution(labels): 
    for i in range(11): 
        print(f'Number of samples for label{i} = {labels.count(i)}')
        
def split_train_and_test(imageurls, labels, numtest): 
    trainX, trainy = imageurls[numtest:], labels[numtest:]
    testX, testy = imageurls[:numtest], labels[:numtest]
    return trainX, testX, trainy, testy

def makePredictionsAndComputeAccuracyDatabase(test_X, test_y):
    predictions =[]
    results = model.predict(test_X)
    for r in results: 
        maxval = np.max(r)
        predictions.append(np.squeeze(np.where(r==maxval)[0]).item(0))
    misclassified = [(i,j) for i, j in zip(predictions, test_y) if i != j]
    print(len([i for i, j in zip(predictions, test_y) if i == j])/len(test_y))
    return misclassified