import psycopg2
import numpy as np
from sklearn.utils import shuffle

def makeHerokuDBConnection(connection_file):
    file = open(connection_file, 'r')
    connection_string = file.readline()
    conn = psycopg2.connect(connection_string.strip())
    curr = conn.cursor()
    return curr

def fetchHerokuData(curr):
    # user count 
    curr.execute('SELECT COUNT(*) FROM users')
    num_users = curr.fetchone()

    # house count
    curr.execute('SELECT COUNT(*) FROM houses')
    num_houses = curr.fetchone()

    # users, houses, scores
    curr.execute('SELECT user_id, house_id, actual_score from matches where actual_score is not null')
    data = curr.fetchall()
    data = shuffle(data, random_state=40)
    data = [list(d) for d in zip(*data)]
    users, houses, matches = data[0], data[1], data[2]
    return users, houses, matches, num_users, num_houses

def makePredictionsAndComputeAccuracy(model, test_X, test_y):
    res = model.predict(test_X)
    res = [ 1 if r >= 0 else 0 for r in res ]
    print(len([i for i, j in zip(test_y, res) if i == j])/len(test_y))

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

def makePredictionsAndComputeAccuracyDatabase(model, test_X, test_y):
    predictions =[]
    results = model.predict(test_X)
    for r in results: 
        maxval = np.max(r)
        predictions.append(np.squeeze(np.where(r==maxval)[0]).item(0))
    misclassified = [(i,j) for i, j in zip(predictions, test_y) if i != j]
    print(len([i for i, j in zip(predictions, test_y) if i == j])/len(test_y))
    return misclassified