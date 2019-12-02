import urllib.request
import csv

pics = []
with open('houses.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        for item in row[0].split(' '):
            if item:
                pics.append(item)

for item in pics[1:]:
    try:
        urllib.request.urlretrieve(item, "images/" + item[item.rfind("/")+1:])
        print(item)
    except: 
        print(f'Not found: {item}')
