import csv
import pyodbc
import pymorphy2
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim.parsing.preprocessing as prep
from nltk.stem import SnowballStemmer
from nltk.stem.snowball import RussianStemmer
import binascii
import operator
import math
from sklearn.neighbors import KNeighborsClassifier
import numpy
from sklearn.metrics import precision_score,recall_score,f1_score,jaccard_similarity_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import time
from PIL import Image
import imagehash
import cv2
import scipy as sp
import sys
from matplotlib import pyplot as plt
from ImageWorker import ImageWorker
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

def showCategory():
    FILENAME = "E:\Study\Diploma\Avito duplicates\Category.csv\Category.csv"
    with open(FILENAME, encoding='utf-8', mode = "r", newline="") as file:
        reader = csv.reader(file)
        i=0
        for row in reader:
            if i < 100:
                print(row)
                i+=1
            else:
                break
def showItemInfo():
    FILENAME = "E:\Study\Diploma\Avito duplicates\ItemInfo_train.csv_2\ItemInfo_train.csv"
    with open(FILENAME, encoding='utf-8', mode = "r", newline="") as file:
        reader = csv.reader(file)
        i=0
        for row in reader:
            if i < 100:
                print(row)
                i+=1
            else:
                break

def writeCategory(cursor):
    FILENAME = "E:\Study\Diploma\Avito duplicates\Category.csv\Category.csv"
    insertQuery = "INSERT INTO Category (categoryID, parentCategoryID) VALUES (?,?);"
    with open(FILENAME, encoding='utf-8', mode = "r", newline="") as file:
        reader = csv.reader(file)
        i=0
        for row in reader:
            if i==0:
                i+=1
                continue
            catID = int(row[0])
            pcatID = int(row[1])
            with cursor.execute(insertQuery, catID, pcatID):
                print ('Successfully Inserted!')

def writeItemInfo(cursor, conn):
    dropQuery = 'drop table if exists ItemInfo'
    createTableQuery = 'create table ItemInfo( itemID int,' \
                       ' categoryID int references Category(categoryID) on delete cascade, title nvarchar(1000),' \
                       'description nvarchar(max),' \
                       'images_array nvarchar(1000),' \
                       'attrsJSON nvarchar(max),' \
                       'price float,' \
                       'locationID int,' \
                       'metroID int,' \
                       'lat decimal(9,6),' \
                       'lon decimal(9,6))'
    with cursor.execute(dropQuery):
        print("ItemInfo dropped")
    with cursor.execute(createTableQuery):
        print("Table created")
    FILENAME = "E:\Study\Diploma\Avito duplicates\ItemInfo_train.csv_2\ItemInfo_train.csv"
    insertQuery = "INSERT INTO ItemInfo (itemID, categoryID, title, description, images_array, attrsJSON," \
                  "price, locationID, metroID, lat, lon ) VALUES (?,?,?,?,?,?,?,?,?,?,?);"
    with open(FILENAME, encoding='utf-8', mode = "r", newline="") as file:
        reader = csv.reader(file)
        i=0
        data = []
        for row in reader:
            if i==0:
                i+=1
                continue
            itemID = categoryID = title = description=images_array=attrsJSON=price=locationID=metroID=lat=lon = None
            try:
                itemID = int(row[0])
                categoryID = int(row[1])
                title = row[2]
                description = row[3]
                images_array = row[4]
                attrsJSON = row[5]
                if row[6] == '':
                    price = 0
                else:
                    price = float(row[6])
                locationID = int(row[7])
                if row[8] == '':
                    metroID = None
                else:
                    metroID = int(float(row[8]))
                lat = float(row[9])
                lon = float(row[10])
                data.append((itemID,categoryID,title,description,images_array,attrsJSON,price,locationID,metroID,lat,lon))
                if i % 10000 == 0:
                    cursor.executemany(insertQuery,data)
                    conn.commit()
                    data=[]
            except ValueError:
                print("Value error")
                print("Price = ",row[6])
                print("ItemID = ",row[0])
                with cursor.execute(insertQuery, itemID, categoryID, title,description,images_array,attrsJSON,
                                    price,locationID,metroID,lat,lon):
                    print ('Inserted in exeption!')
            finally:
                i+=1
        cursor.executemany(insertQuery,data)
        conn.commit()
        conn.close()

def writeDuplicate(cursor):
    selectQuery = 'select top 1000 itemID_1,itemID_2 from ItemPairs where isDuplicate=1'
    itemID = []
    with cursor.execute(selectQuery):
        row = cursor.fetchone()
        while row:
            itemID.append((row[0],row[1]))
            row = cursor.fetchone()

    dropQuery = 'drop table if exists Duplicate'
    createTableQuery = 'create table Duplicate(' \
                       'id int identity(1,1),'\
                       'itemID_1 int,' \
                       'categoryID_1 int,' \
                       ' title_1 nvarchar(1000),' \
                       'description_1 nvarchar(max),' \
                       'images_array_1 nvarchar(1000),' \
                       'attrsJSON_1 nvarchar(max),' \
                       'price_1 float,' \
                       'locationID_1 int,' \
                       'metroID_1 int,' \
                       'lat_1 decimal(9,6),' \
                       'lon_1 decimal(9,6),' \
                        'itemID_2 int,' \
                        'categoryID_2 int,' \
                       ' title_2 nvarchar(1000),' \
                        'description_2 nvarchar(max),' \
                        'images_array_2 nvarchar(1000),' \
                        'attrsJSON_2 nvarchar(max),' \
                        'price_2 float,' \
                        'locationID_2 int,' \
                        'metroID_2 int,' \
                        'lat_2 decimal(9,6),' \
                        'lon_2 decimal(9,6))'
    with cursor.execute(dropQuery):
        print("Duplicate dropped")
    with cursor.execute(createTableQuery):
        print("Duplicate created")

    insertQuery = 'insert into Duplicate(itemID_1,categoryID_1,title_1,description_1,' \
                  'images_array_1,attrsJSON_1,price_1,locationID_1,metroID_1,lat_1,lon_1,' \
                  'itemID_2,categoryID_2,title_2,description_2,' \
                  'images_array_2,attrsJSON_2,price_2,locationID_2,metroID_2,lat_2,lon_2)' \
                  'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
    selectQuery = 'select * from ItemInfo where itemID in (?,?)'
    data = []
    for id in itemID:
        with cursor.execute(selectQuery, id):
            row = cursor.fetchone()
            data.extend(row)
            row = cursor.fetchone()
            data.extend(row)
        with cursor.execute(insertQuery,data):
            data=[]
            print("Data inserted")

def writeDuplicateGenM(cursor, genMethod, size=1000):
    selectQuery = 'select top ' + str(size) + ' itemID_1,itemID_2 from ItemPairs where isDuplicate=1 and generationMethod= ' + str(genMethod)
    itemID = []
    with cursor.execute(selectQuery):
        row = cursor.fetchone()
        while row:
            itemID.append((row[0],row[1]))
            row = cursor.fetchone()

    dropQuery = 'drop table if exists DuplicateGenM' + str(genMethod) + "_" + str(size)
    createTableQuery = 'create table DuplicateGenM' + str(genMethod) + "_" + str(size) + '(' \
                       'id int identity(1,1),' \
                       'itemID_1 int,' \
                       'categoryID_1 int,' \
                       ' title_1 nvarchar(1000),' \
                       'description_1 nvarchar(max),' \
                       'images_array_1 nvarchar(1000),' \
                       'attrsJSON_1 nvarchar(max),' \
                       'price_1 float,' \
                       'locationID_1 int,' \
                       'metroID_1 int,' \
                       'lat_1 decimal(9,6),' \
                       'lon_1 decimal(9,6),' \
                       'itemID_2 int,' \
                       'categoryID_2 int,' \
                       ' title_2 nvarchar(1000),' \
                       'description_2 nvarchar(max),' \
                       'images_array_2 nvarchar(1000),' \
                       'attrsJSON_2 nvarchar(max),' \
                       'price_2 float,' \
                       'locationID_2 int,' \
                       'metroID_2 int,' \
                       'lat_2 decimal(9,6),' \
                       'lon_2 decimal(9,6))'
    with cursor.execute(dropQuery):
        print("Duplicate dropped")
    with cursor.execute(createTableQuery):
        print("Duplicate created")

    insertQuery = 'insert into DuplicateGenM' + str(genMethod) + "_" + str(size) + '(itemID_1,categoryID_1,title_1,description_1,' \
                  'images_array_1,attrsJSON_1,price_1,locationID_1,metroID_1,lat_1,lon_1,' \
                  'itemID_2,categoryID_2,title_2,description_2,' \
                  'images_array_2,attrsJSON_2,price_2,locationID_2,metroID_2,lat_2,lon_2)' \
                  'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
    selectQuery = 'select * from ItemInfo where itemID in (?,?)'
    data = []
    for id in itemID:
        with cursor.execute(selectQuery, id):
            row = cursor.fetchone()
            data.extend(row)
            row = cursor.fetchone()
            data.extend(row)
        with cursor.execute(insertQuery,data):
            data=[]
            print("Data inserted")

def writeNotDuplicate(cursor):
    selectQuery = 'select top 1000 itemID_1,itemID_2 from ItemPairs where isDuplicate=0'
    itemID = []
    with cursor.execute(selectQuery):
        row = cursor.fetchone()
        while row:
            itemID.append((row[0],row[1]))
            row = cursor.fetchone()

    dropQuery = 'drop table if exists NoDuplicate'
    createTableQuery = 'create table NoDuplicate(' \
                       'id int identity(1,1),' \
                       'itemID_1 int,' \
                       'categoryID_1 int,' \
                       ' title_1 nvarchar(1000),' \
                       'description_1 nvarchar(max),' \
                       'images_array_1 nvarchar(1000),' \
                       'attrsJSON_1 nvarchar(max),' \
                       'price_1 float,' \
                       'locationID_1 int,' \
                       'metroID_1 int,' \
                       'lat_1 decimal(9,6),' \
                       'lon_1 decimal(9,6),' \
                       'itemID_2 int,' \
                       'categoryID_2 int,' \
                       ' title_2 nvarchar(1000),' \
                       'description_2 nvarchar(max),' \
                       'images_array_2 nvarchar(1000),' \
                       'attrsJSON_2 nvarchar(max),' \
                       'price_2 float,' \
                       'locationID_2 int,' \
                       'metroID_2 int,' \
                       'lat_2 decimal(9,6),' \
                       'lon_2 decimal(9,6))'
    with cursor.execute(dropQuery):
        print("NoDuplicate dropped")
    with cursor.execute(createTableQuery):
        print("NoDuplicate created")

    insertQuery = 'insert into NoDuplicate(itemID_1,categoryID_1,title_1,description_1,' \
                  'images_array_1,attrsJSON_1,price_1,locationID_1,metroID_1,lat_1,lon_1,' \
                  'itemID_2,categoryID_2,title_2,description_2,' \
                  'images_array_2,attrsJSON_2,price_2,locationID_2,metroID_2,lat_2,lon_2)' \
                  'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
    selectQuery = 'select * from ItemInfo where itemID in (?,?)'
    data = []
    for id in itemID:
        with cursor.execute(selectQuery, id):
            row = cursor.fetchone()
            data.extend(row)
            row = cursor.fetchone()
            data.extend(row)
        with cursor.execute(insertQuery,data):
            data=[]
            print("Data inserted")

def writeNoDuplicateGenM(cursor, genMethod, size=1000):
    selectQuery = 'select top ' + str(size) + ' itemID_1,itemID_2 from ItemPairs where isDuplicate=0 and generationMethod=' + str(genMethod)
    itemID = []
    with cursor.execute(selectQuery):
        row = cursor.fetchone()
        while row:
            itemID.append((row[0],row[1]))
            row = cursor.fetchone()

    dropQuery = 'drop table if exists NoDuplicateGenM' + str(genMethod) + '_'  + str(size)
    createTableQuery = 'create table NoDuplicateGenM' + str(genMethod) + '_'  + str(size) + '(' \
                                                                      'id int identity(1,1),' \
                                                                      'itemID_1 int,' \
                                                                      'categoryID_1 int,' \
                                                                      ' title_1 nvarchar(1000),' \
                                                                      'description_1 nvarchar(max),' \
                                                                      'images_array_1 nvarchar(1000),' \
                                                                      'attrsJSON_1 nvarchar(max),' \
                                                                      'price_1 float,' \
                                                                      'locationID_1 int,' \
                                                                      'metroID_1 int,' \
                                                                      'lat_1 decimal(9,6),' \
                                                                      'lon_1 decimal(9,6),' \
                                                                      'itemID_2 int,' \
                                                                      'categoryID_2 int,' \
                                                                      ' title_2 nvarchar(1000),' \
                                                                      'description_2 nvarchar(max),' \
                                                                      'images_array_2 nvarchar(1000),' \
                                                                      'attrsJSON_2 nvarchar(max),' \
                                                                      'price_2 float,' \
                                                                      'locationID_2 int,' \
                                                                      'metroID_2 int,' \
                                                                      'lat_2 decimal(9,6),' \
                                                                      'lon_2 decimal(9,6))'
    with cursor.execute(dropQuery):
        print("Duplicate dropped")
    with cursor.execute(createTableQuery):
        print("Duplicate created")

    insertQuery = 'insert into NoDuplicateGenM' + str(genMethod) + '_'  + str(size) + '(itemID_1,categoryID_1,title_1,description_1,' \
                                                             'images_array_1,attrsJSON_1,price_1,locationID_1,metroID_1,lat_1,lon_1,' \
                                                             'itemID_2,categoryID_2,title_2,description_2,' \
                                                             'images_array_2,attrsJSON_2,price_2,locationID_2,metroID_2,lat_2,lon_2)' \
                                                             'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
    selectQuery = 'select * from ItemInfo where itemID in (?,?)'
    data = []
    for id in itemID:
        with cursor.execute(selectQuery, id):
            row = cursor.fetchone()
            data.extend(row)
            row = cursor.fetchone()
            data.extend(row)
        with cursor.execute(insertQuery,data):
            data=[]
            print("Data inserted")

def writeItemPairs(cursor):
    dropQuery = 'drop table if exists ItemPairs'
    createTableQuery = 'create table ItemPairs( itemID_1 int,' \
                       ' itemID_2 int,' \
                       'isDuplicate tinyint,' \
                       'generationMethod tinyint)'
    with cursor.execute(dropQuery):
        print("ItemPairs dropped")
    with cursor.execute(createTableQuery):
        print("Table created")

    FILENAME = "E:\Study\Diploma\Avito duplicates\ItemPairs_train.csv_2\ItemPairs_train.csv"
    insertQuery = "INSERT INTO ItemPairs (itemID_1, itemID_2, isDuplicate, generationMethod)" \
                  " VALUES (?,?,?,?);"
    with open(FILENAME, encoding='utf-8', mode = "r", newline="") as file:
        reader = csv.reader(file)
        i=0
        for row in reader:
            if i==0:
                i+=1
                continue

            itemID_1 = int(row[0])
            itemID_2 = int(row[1])
            isDuplicate = int(row[2])
            generationMethod = int(row[3])

            with cursor.execute(insertQuery, itemID_1, itemID_2, isDuplicate,generationMethod):
                print ('Successfully Inserted!')

def test_normalization(text1,text2):
    morph = pymorphy2.MorphAnalyzer()
    tokenizer = RegexpTokenizer(r'[^\W\d_]+|\d+')
    stemmer = SnowballStemmer(language='russian',ignore_stopwords=True)
    stop_words = stopwords.words('russian')
    prep_text1 = prep.stem_text(prep.strip_short(text1))
    tokens = tokenizer.tokenize(prep_text1)
    tokens = [i.lower() for i in tokens if ( i not in stop_words and len(i)>3 )]
    stems = [stemmer.stem(word)[:4] for word in tokens]
    # morphs = [morph.parse(token)[0].normal_form for token in tokens ]
    print('stems',stems)

    tokens = tokenizer.tokenize(text2)
    tokens = [i.lower() for i in tokens if ( i not in stop_words and len(i)>3 )]
    # morphs = [morph.parse(token)[0].normal_form for token in tokens ]
    stems = [stemmer.stem(word)[:4] for word in tokens]
    print('stems',stems)

def textToVec(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'[^\W\d_]+|\d+')
    stop_words = stopwords.words('russian')
    stemmer = SnowballStemmer(language='russian',ignore_stopwords=True)
    morph = pymorphy2.MorphAnalyzer()
    tokens = tokenizer.tokenize(text)
    digits = []
    for i in range(len(tokens)):
        try:
            token = tokens[i]
            if token.isdigit():
                float(token) # test digit
                digits.append(token)
        except ValueError:
            print("ValueError has happened")
    tokens = [i for i in tokens if (i not in stop_words and len(i)>=3 )]
    stems = [stemmer.stem(word)[:3] for word in tokens]
    morphs = [morph.parse(token)[0].normal_form for token in tokens if token not in stop_words and len(token)>=3 ]
    stems.extend(digits)
    morphs.extend(digits)
    return stems

def dot_product2(v1, v2):
    return sum(map(operator.mul, v1, v2))

def computeDistance(text1, text2):
    words = []
    tokenizer = RegexpTokenizer(r'[^\W\d_]+|\d+')
    stemmer = SnowballStemmer(language='russian',ignore_stopwords=True)
    stop_words = stopwords.words('russian')
    tokens = tokenizer.tokenize(text1)
    tokens = [i for i in tokens if ( i not in stop_words and len(i)>3 )]
    stems1 = [stemmer.stem(word)[:4] for word in tokens]
    words.extend(stems1)
    tokens = tokenizer.tokenize(text2)
    tokens = [i for i in tokens if ( i not in stop_words and len(i)>3 )]
    stems2 = [stemmer.stem(word)[:4] for word in tokens]
    words.extend(stems2)
    words = set(words)
    v1 = []
    v2 = []
    for word in words:
        if word in stems1:
            v1.append(1)
        else:
            v1.append(0)
        if word in stems2:
            v2.append(1)
        else:
            v2.append(0)

    prod = dot_product2(v1, v2)
    len1 = math.sqrt(dot_product2(v1, v1))
    len2 = math.sqrt(dot_product2(v2, v2))
    return prod / (len1 * len2)

def readDuplicates(cursor, limit):
    selectQuery = 'select top ' + str(limit) + ' id, description_1, description_2 from Duplicate'
    data = []
    with cursor.execute(selectQuery):
        row = cursor.fetchone()
        while row:
            id = row[0]
            desc_1 = row[1]
            desc_2 = row[2]
            data.append((id,desc_1,desc_2))
            row = cursor.fetchone()
    return data

def readAttrs(cursor, limit, tableName):
    selectQuery = 'select top ' + str(limit) + ' id, attrsJSON_1, attrsJSON_2 from ' + tableName
    data = []
    with cursor.execute(selectQuery):
        row = cursor.fetchone()
        while row:
            id = row[0]
            attrs_1 = row[1]
            attrs_2 = row[2]
            data.append((id,attrs_1,attrs_2))
            row = cursor.fetchone()
    return data

def readNoDuplicates(cursor, limit):
    selectQuery = 'select top ' + str(limit) + ' id, description_1, description_2 from NoDuplicate'
    data = []
    with cursor.execute(selectQuery):
        row = cursor.fetchone()
        while row:
            id = row[0]
            desc_1 = row[1]
            desc_2 = row[2]
            data.append((id,desc_1,desc_2))
            row = cursor.fetchone()
    return data

def prepareDuplicates(data):
    vec=[]
    for row in data:
        id = row[0]
        t1 = row[1]
        v1 = textToVec(t1)
        t2 = row[2]
        v2 = textToVec(t2)
        vec.append((id,v1,v2))

    return vec

def computeSimilarity(words1,words2):
    words=[]
    words.extend(words1)
    words.extend(words2)
    words=set(words)
    v1 = []
    v2 = []
    for word in words:
        if word in words1:
            v1.append(1)
        else:
            v1.append(0)
        if word in words2:
            v2.append(1)
        else:
            v2.append(0)
    prod = dot_product2(v1, v2)
    v1_dot_prod = dot_product2(v1, v1)
    v2_dot_prod = dot_product2(v2, v2)
    if v1_dot_prod == v2_dot_prod==prod:
        return 1
    else:
        len1 = math.sqrt(dot_product2(v1, v1))
        len2 = math.sqrt(dot_product2(v2, v2))
        if len1==0 or len2==0: return 0
    return prod / (len1 * len2)

def computeJakkarSimilarity(words1,words2):
    words=[]
    words.extend(words1)
    words.extend(words2)
    words = set(words)
    v1 = []
    v2 = []
    for word in words:
        if word in words1:
            v1.append(1)
        else:
            v1.append(0)
        if word in words2:
            v2.append(1)
        else:
            v2.append(0)
    if(len(words)==0):
        return 0
    return jaccard_similarity_score(v1,v2)

def computeShinglesJakkarSimilarity(words1, words2, shinglesLen):
    text1 = ''
    text2 = ''
    for shingle in words1:
        text1 = text1 + shingle
    for shingle in words2:
        text2 = text2 + shingle

    shingles1 = []
    for i in range(0, len(text1) - shinglesLen):
        shingles1.append(text1[i:i+shinglesLen])

    shingles2 = []
    for i in range(0, len(text2) - shinglesLen):
        shingles2.append(text2[i:i+shinglesLen])

    shingles = []
    shingles.extend(shingles1)
    shingles.extend(shingles2)
    shingles = set(shingles)
    v1 = []
    v2 = []
    for shingle in shingles:
        if shingle in shingles1:
            v1.append(1)
        else:
            v1.append(0)
        if shingle in shingles2:
            v2.append(1)
        else:
            v2.append(0)
    if(len(shingles)==0):
        return 0
    return jaccard_similarity_score(v1,v2)


def checkDuplicates(vectors):
    sim = []
    for row in vectors:
        id = row[0]
        v1 = row[1]
        v2 = row[2]
        similarity = computeJakkarSimilarity(v1,v2)
        sim.append((id,v1, v2, similarity))
    return sim

def computePercent(res):
    sum = 0
    for value in res:
        if value[3]>=0.5:
           sum+=1

    return sum/len(res)

def readData(cursor, limit, generationMethod):
    if generationMethod != 0:
        selectQuery = 'select top ' + str(limit) + ' itemID_1,itemID_2,isDuplicate from ItemPairs where generationMethod= ' + str(generationMethod)
    else:
        selectQuery = 'select top ' + str(limit) + ' itemID_1,itemID_2,isDuplicate from ItemPairs'
    itemID = []
    Y = []
    with cursor.execute(selectQuery):
        row = cursor.fetchone()
        while row:
            itemID.append((row[0],row[1]))
            Y.append(row[2])
            row = cursor.fetchone()

    selectQuery = 'select * from ItemInfo where itemID in (?,?)'
    X = []
    for pair in itemID:
        with cursor.execute(selectQuery,pair):
            row1 = cursor.fetchone()
            row2 = cursor.fetchone()
            X.append((row1,row2))

    data = (X,Y)
    return data

def writeMixedDataM1_20000(cursor):
    duplicates_query = "select * from DuplicateGenM1_10000"
    not_duplicates_query = "select * from NoDuplicateGenM1_10000"
    mixedData = []
    duplicatesData = []
    notDuplicatesData = []
    with cursor.execute(duplicates_query):
        row = cursor.fetchone()
        while row:
            duplicatesData.append(row[1:])
            row = cursor.fetchone()
    with cursor.execute(not_duplicates_query):
        row = cursor.fetchone()
        while row:
            notDuplicatesData.append(row[1:])
            row = cursor.fetchone()

    for i in range(10000):
            item = []
            item.append(1) # duplicate label
            item.extend(duplicatesData[i])
            mixedData.append(item)
            item = []
            item.append(0) # not duplicate label
            item.extend(notDuplicatesData[i])
            mixedData.append(item)
    print("Mixed data shape", numpy.array(mixedData).shape)
    dropQuery = 'drop table if exists MixedData'
    createTableQuery = 'create table MixedData(' \
                       'id int PRIMARY KEY identity(1,1),' \
                       'Y int,' \
                       'itemID_1 int,' \
                       'categoryID_1 int,' \
                       'title_1 nvarchar(1000),' \
                       'description_1 nvarchar(max),' \
                       'images_array_1 nvarchar(1000),' \
                       'attrsJSON_1 nvarchar(max),' \
                       'price_1 float,' \
                       'locationID_1 int,' \
                       'metroID_1 int,' \
                       'lat_1 decimal(9,6),' \
                       'lon_1 decimal(9,6),' \
                       'itemID_2 int,' \
                       'categoryID_2 int,' \
                       'title_2 nvarchar(1000),' \
                       'description_2 nvarchar(max),' \
                       'images_array_2 nvarchar(1000),' \
                       'attrsJSON_2 nvarchar(max),' \
                       'price_2 float,' \
                       'locationID_2 int,' \
                       'metroID_2 int,' \
                       'lat_2 decimal(9,6),' \
                       'lon_2 decimal(9,6))'

    with cursor.execute(dropQuery):
        print("Table dropped")
    with cursor.execute(createTableQuery):
        print("Table created")

    insertQuery = 'insert into MixedData' + \
                  '(' \
                                            'Y, ' \
                                            'itemID_1,' \
                                            'categoryID_1,' \
                                            'title_1,' \
                                            'description_1,' \
                                          'images_array_1,' \
                                            'attrsJSON_1,' \
                                            'price_1,' \
                                            'locationID_1,' \
                                            'metroID_1,' \
                                            'lat_1,' \
                                            'lon_1,' \
                                          'itemID_2,' \
                                            'categoryID_2,' \
                                            'title_2,' \
                                            'description_2,' \
                                          'images_array_2,' \
                                            'attrsJSON_2,' \
                                            'price_2,' \
                                            'locationID_2,' \
                                            'metroID_2,' \
                                            'lat_2,' \
                                            'lon_2)' \
                                          'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'

    for i in range(2*10000):
        with cursor.execute(insertQuery, mixedData[i]):
            print("Data inserted")

def readMixedDataNew(cursor, limit):
    selectQuery = "select top " + str(limit) + " * from MixedData"
    X = []
    Y = []
    with cursor.execute(selectQuery):
        row = cursor.fetchone()
        while row:
            Y.append(row[1])
            x1 = row[2:13]
            x2 = row[13:]
            X.append((x1,x2))
            row = cursor.fetchone()
    return X, Y

# read duplicate and not duplicate together
def readMixedData(cursor, limit, generationMethod):
    if generationMethod != 0:
        selectQuery = 'select top ' + str(int(limit / 2)) + ' itemID_1,itemID_2,isDuplicate from ItemPairs where isDuplicate=1 and generationMethod= ' + str(generationMethod) + \
                      'union select top ' + str(int(limit / 2)) + ' itemID_1,itemID_2,isDuplicate from ItemPairs where isDuplicate=0 and generationMethod= ' + str(generationMethod)
    else:
        selectQuery = 'select top ' + str(limit) + ' itemID_1,itemID_2,isDuplicate from ItemPairs'
    itemID = []
    Y = []
    with cursor.execute(selectQuery):
        row = cursor.fetchone()
        while row:
            itemID.append((row[0],row[1]))
            Y.append(row[2])
            row = cursor.fetchone()

    selectQuery = 'select * from ItemInfo where itemID in (?,?)'
    X = []
    for pair in itemID:
        with cursor.execute(selectQuery, pair):
            row1 = cursor.fetchone()
            row2 = cursor.fetchone()
            X.append((row1,row2))

    data = (X,Y)
    return data

def compareNumbers(ad1, ad2):
    numbers1 = [i for i in ad1 if (i.isdigit())]
    numbers2 = [i for i in ad2 if (i.isdigit())]
    return computeJakkarSimilarity(numbers1,numbers2)

def readFeatures(cursor, data_size):
    select_query = "select top " + str(data_size) + " * from Features"
    X = []
    Y = []
    with cursor.execute(select_query):
        data = cursor.fetchall()
        for row in data:
            Y.append(row[1])
            X.append(list(row[4:]))

    result = (X,Y)
    return result

def readNewFeatures(cursor, data_size):
    select_query = "select top " + str(data_size) + " * from FeaturesNew"
    X = []
    Y = []
    with cursor.execute(select_query):
        data = cursor.fetchall()
        for row in data:
            Y.append(row[1])
            X.append(row[4:])

    X = preprocessing.normalize(X, norm='max', axis=0)

    result = (X,Y)
    return result

def readTextFeatures(cursor, data_size):
    select_query = "select top " + str(data_size) + " * from FeaturesNew"
    X = []
    Y = []
    with cursor.execute(select_query):
        data = cursor.fetchall()
        for row in data:
            Y.append(row[1])
            tmp = list(row[4:20])
            tmp.extend(row[24:])
            X.append(tmp)

    X = preprocessing.normalize(X, norm='max', axis=0)

    result = (X,Y)
    return result

def writeFeaturesNew(cursor, data, hash=0, shinglesLen=0):
    tableName = "FeaturesNew"
    dropQuery = 'drop table if exists '  + tableName
    createTableQuery = 'create table ' + tableName + '(' \
                       'id int identity(1,1),' \
                       'Y int,' \
                       'id1 int,' \
                       'id2 int,' \
                       'constraint PK_FEATURES_NEW PRIMARY KEY (id1, id2),' \
                       'cat int,' \
                       'title float,' \
                       'titNumbLen int,' \
                       'titNumbSim int,' \
                       'titNumbMed int,' \
                       'descr float,' \
                       'descNumbLen int,' \
                       'descNumbSim int,' \
                       'descNumbMed int,' \
                       'title1desc2 float,' \
                       'title2desc1 float,' \
                       'titleDesc float,' \
                       'titDescNumbLen int,' \
                       'titDescNumbSim int,' \
                       'titDescNumbMed int,' \
                       'imgNumb int,' \
                       'ahash int,' \
                       'phash int,' \
                       'dhash int, ' \
                       'hist int,' \
                       'attrs float,' \
                       'price int,' \
                       'priceDif float,' \
                       'loc int,' \
                       'metro int,' \
                       'lat int,' \
                       'lon int,' \
                       'latDif float,' \
                       'lonDif float)'

    with cursor.execute(dropQuery):
        print("Table have been dropped")
    with cursor.execute(createTableQuery):
        print("Table have been created")

    insertQuery = 'insert into ' + tableName +'(Y, id1, id2, cat, title, titNumbLen, titNumbSim, titNumbMed,' \
                                              'descr, descNumbLen, descNumbSim, descNumbMed,' \
                                              ' title1desc2, title2desc1, titleDesc, titDescNumbLen,' \
                                              'titDescNumbSim, titDescNumbMed, imgNumb, ahash, phash, dhash,' \
                                              'hist, attrs, price, priceDif, loc, metro, lat, lon, latDif, lonDif)' \
                  ' values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
    pairs = data[0]
    Y = data[1]
    insert_values = []
    for i in range(len(Y)):
        y = Y[i]
        first = pairs[i][0]

        second = pairs[i][1]
        item = []
        item.append(y)
        item.append(first[0])
        print("itemID_1 = ", first[0])
        item.append(second[0])
        print("itemID_2 = ", second[0])

        cat1 = first[1]
        cat2 = second[1]
        if cat1==cat2:
            item.append(1)
        else:
            item.append(0)
        if hash == 1:
            title1 = wordsToNumbers(textToVec(first[2]))
            title2 = wordsToNumbers(textToVec(second[2]))
        else:
            title1 = textToVec(first[2])
            title2 = textToVec(second[2])
        if shinglesLen == 0:
            titleSimilarity = computeJakkarSimilarity(title1,title2)
        else:
            titleSimilarity = computeShinglesJakkarSimilarity(title1,title2,shinglesLen)
        item.append(titleSimilarity)

        # Does the number of numbers in the titles match?
        numbers1 = [float(i) for i in title1 if (i.isdigit())]
        numbers2 = [float(i) for i in title2 if (i.isdigit())]
        if len(numbers1) == 0 and len(numbers2) == 0:
            item.append(1) # lengths equals
            item.append(1) # sim=1
            item.append(1) # median = 1
        elif len(numbers1) == 0 or len(numbers2) == 0:
            item.append(0)
            item.append(0)
            item.append(0)
        else:
            item.append(int(len(numbers1) == len(numbers2)))
            # the percentage of matching numbers
            title_numbers_similarity = compareNumbers(title1, title2)
            item.append(title_numbers_similarity)
            # median
            item.append(int(numpy.median(numbers1) == numpy.median(numbers2)))

        if hash == 1:
            description1 = wordsToNumbers(textToVec(first[3]))
            description2 = wordsToNumbers(textToVec(second[3]))
        else:
            description1 = textToVec(first[3])
            description2 = textToVec(second[3])
        if shinglesLen == 0:
            descriptionSimilarity = computeJakkarSimilarity(description1,description2)
        else:
            descriptionSimilarity = computeShinglesJakkarSimilarity(description1,description2,shinglesLen)
        item.append(descriptionSimilarity)

        # Does the number of numbers in the descriptions match?
        numbers1 = [float(i) for i in description1 if (i.isdigit())]
        numbers2 = [float(i) for i in description2 if (i.isdigit())]
        if len(numbers1) == 0 and len(numbers2) == 0:
            item.append(1) # lengths equals
            item.append(1) # sim=1
            item.append(1) # median = 1
        elif len(numbers1) == 0 or len(numbers2) == 0:
            item.append(0)
            item.append(0)
            item.append(0)
        else:
            item.append(int(len(numbers1) == len(numbers2)))
            # the percentage of matching numbers
            title_numbers_similarity = compareNumbers(title1, title2)
            item.append(title_numbers_similarity)
            # median
            item.append(int(numpy.median(numbers1) == numpy.median(numbers2)))
        # compare title vs description and so on
        if shinglesLen == 0:
            title1Desc2 = computeJakkarSimilarity(title1,description2)
            title2Desc1 = computeJakkarSimilarity(title2, description1)
            item.append(title1Desc2)
            item.append(title2Desc1)
        else:
            title1Desc2 = computeShinglesJakkarSimilarity(title1, description2,shinglesLen)
            title2Desc1 = computeShinglesJakkarSimilarity(title2, description1,shinglesLen)
            item.append(title1Desc2)
            item.append(title2Desc1)
        title1.extend(description1)
        title2.extend(description2)
        if shinglesLen == 0:
            titleDescSim = computeJakkarSimilarity(title1,title2)
        else:
            titleDescSim = computeShinglesJakkarSimilarity(title1,title2,shinglesLen)
        item.append(titleDescSim)

        # Does the number of numbers in the title + description match?
        numbers1 = [float(i) for i in title1 if (i.isdigit())]
        numbers2 = [float(i) for i in title2 if (i.isdigit())]

        if len(numbers1) == 0 and len(numbers2) == 0:
            item.append(1) # lengths equals
            item.append(1) # sim=1
            item.append(1) # median = 1
        elif len(numbers1) == 0 or len(numbers2) == 0:
            item.append(0)
            item.append(0)
            item.append(0)
        else:
            item.append(int(len(numbers1) == len(numbers2)))
            # the percentage of matching numbers
            title_numbers_similarity = compareNumbers(title1, title2)
            item.append(title_numbers_similarity)
            # median
            item.append(int(numpy.median(numbers1) == numpy.median(numbers2)))

        images_str1 = first[4]
        images_str2 = second[4]
        if images_str1 != '' and images_str2 != '':
            img_nums1 = images_str1.split(', ')
            img_nums2 = images_str2.split(', ')
            paths1 = create_paths(img_nums1)
            paths2 = create_paths(img_nums2)
            # Is number of images equals?
            if len(img_nums1) == len(img_nums2):
                item.append(1)
            else:
                item.append(0)
            computeHashes(item, paths1, paths2, 'ahash')
            computeHashes(item, paths1, paths2, 'phash')
            computeHashes(item, paths1, paths2, 'dhash')
            images1 = get_images(img_nums1)
            images2 = get_images(img_nums2)
            histograms1 = compute_histograms(images1, type='lab', hist_size=32)
            histograms2 = compute_histograms(images2, type='lab', hist_size=32)
            hist_f = compare_histograms(histograms1, histograms2, method="hellinger")
            item.append(hist_f)
        elif images_str1 != '' or images_str2 != '':
            item.append(0) # number of images doesn't equals
            # three types of hashes
            # 0 because not one images
            item.append(0)
            item.append(0)
            item.append(0)
            # not one image not histograms
            item.append(0)
        else:
            item.append(1) # number of images equals and = 0
            # three types of hashes
            item.append(1)
            item.append(1)
            item.append(1)
            item.append(1)

        if hash == 1:
            attrsJSON1 = wordsToNumbers(textToVec(first[5]))
            attrsJSON2 = wordsToNumbers(textToVec(second[5]))
        else:
            attrsJSON1 = textToVec(first[5])
            attrsJSON2 = textToVec(second[5])
        if shinglesLen == 0:
            attrsSim = computeJakkarSimilarity(attrsJSON1, attrsJSON2)
        else:
            attrsSim = computeShinglesJakkarSimilarity(title1,title2,shinglesLen)

        item.append(attrsSim)
        price1 = first[6]
        price2 = second[6]
        item.append(int(price1 == price2))

        if price1 == price2:
            item.append(1)
        else:
            item.append(1 / math.fabs(price1-price2))
        location1 = first[7]
        location2 = second[7]
        item.append(int(location1==location2))
        metro1 = first[8]
        metro2 = second[8]
        if metro1 is None and metro2 is None:
            item.append(1) # metroEqual attr
        else:
            if metro1 is None or metro2 is None:
                item.append(0)
            else:
                if metro1 == metro2:
                    item.append(1)
                else:
                    item.append(0)

        lat1 = first[9]
        lat2 = second[9]
        item.append(int(lat1==lat2))
        lon1 = first[10]
        lon2 = second[10]
        item.append(int(lon1==lon2))
        if lon1 == lon2:
            item.append(1) # lon closeness
        else:
            item.append(1 / (math.fabs(lon1-lon2)))

        if lat1 == lat2:
            item.append(1) # lat closeness
        else:
            item.append(1 / (math.fabs(lat1-lat2)))

        insert_values.append(tuple(item))

    # with doesn't work
    cursor.executemany(insertQuery, insert_values)
    cursor.commit()
    cursor.close()
    print('insert_values have been inserted')


def writeFeatures(cursor,data,shinglesLen, hash):
    dropQuery = 'drop table if exists Features'
    createTableQuery = 'create table Features(' \
                       'id int identity(1,1),' \
                       'Y int,' \
                       'id1 int,' \
                       'id2 int,' \
                       'constraint PK PRIMARY KEY (id1, id2),' \
                       'cat int,' \
                       'title float,' \
                       'descr float,' \
                       'title1desc2 float,' \
                       'title2desc1 float,' \
                       'titleDesc float,' \
                       'hash int,' \
                       'ahash int,' \
                       'phash int,' \
                       'dhash int, ' \
                       'attrs float,' \
                       'price int,' \
                       'priceDif float,' \
                       'loc int,' \
                       'nonMetro int,' \
                       'metro int,' \
                       'lat int,' \
                       'lon int,' \
                       'latDif float,' \
                       'lonDif float)'

    with cursor.execute(dropQuery):
        print("Features dropped")
    with cursor.execute(createTableQuery):
        print("Features created")

    insertQuery = 'insert into Features(Y, id1, id2, cat, title, descr, title1desc2, title2desc1, titleDesc, hash, ahash, phash, dhash, attrs, price, priceDif, loc, nonMetro, metro, lat, lon, latDif, lonDif)' \
                  ' values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'

    pairs = data[0]
    Y = data[1]
    insert_values = []
    for i in range(len(Y)):
        y = Y[i]
        first = pairs[i][0]
        second = pairs[i][1]
        item = []
        item.append(y)
        item.append(first[0])
        item.append(second[0])
        cat1 = first[1]
        cat2 = second[1]
        if cat1 == cat2:
            item.append(1)
        else:
            item.append(0)
        if hash == 1:
            title1 = wordsToNumbers(textToVec(first[2]))
            title2 = wordsToNumbers(textToVec(second[2]))
        else:
            title1 = textToVec(first[2])
            title2 = textToVec(second[2])
        if shinglesLen == 0:
            titleSimilarity = computeJakkarSimilarity(title1,title2)
        else:
            titleSimilarity = computeShinglesJakkarSimilarity(title1,title2,shinglesLen)
        item.append(titleSimilarity)
        if hash == 1:
            description1 = wordsToNumbers(textToVec(first[3]))
            description2 = wordsToNumbers(textToVec(second[3]))
        else:
            description1 = textToVec(first[3])
            description2 = textToVec(second[3])
        if shinglesLen == 0:
            descriptionSimilarity = computeJakkarSimilarity(title1,title2)
        else:
            descriptionSimilarity = computeShinglesJakkarSimilarity(title1,title2,shinglesLen)
        item.append(descriptionSimilarity)
        # compare title vs description and so on
        if shinglesLen == 0:
            title1Desc2 = computeJakkarSimilarity(title1,description2)
            title2Desc1 = computeJakkarSimilarity(title2, description1)
            item.append(title1Desc2)
            item.append(title2Desc1)
        else:
            title1Desc2 = computeShinglesJakkarSimilarity(title1, description2,shinglesLen)
            title2Desc1 = computeShinglesJakkarSimilarity(title2, description1,shinglesLen)
            item.append(title1Desc2)
            item.append(title2Desc1)
        title1.extend(description1)
        title2.extend(description2)
        if shinglesLen == 0:
            titleDescSim = computeJakkarSimilarity(title1,title2)
        else:
            titleDescSim = computeShinglesJakkarSimilarity(title1,title2,shinglesLen)
        item.append(titleDescSim)

        images_str1 = first[4]
        images_str2 = second[4]
        if images_str1 != '' and images_str2 != '':
            item.append(1) # there are images
            img_arr1 = images_str1.split(', ')
            img_arr2 = images_str2.split(', ')
            computeHashes(item, img_arr1, img_arr2, 'ahash')
            computeHashes(item, img_arr1, img_arr2, 'phash')
            computeHashes(item, img_arr1, img_arr2, 'dhash')
        else:
            item.append(0) # there are not images
            item.append(0)
            item.append(0)
            item.append(0)
        if hash == 1:
            attrsJSON1 = wordsToNumbers(textToVec(first[5]))
            attrsJSON2 = wordsToNumbers(textToVec(second[5]))
        else:
            attrsJSON1 = textToVec(first[5])
            attrsJSON2 = textToVec(second[5])
        if shinglesLen == 0:
            attrsSim = computeJakkarSimilarity(attrsJSON1, attrsJSON2)
        else:
            attrsSim = computeShinglesJakkarSimilarity(title1,title2,shinglesLen)

        item.append(attrsSim)
        price1 = first[6]
        price2 = second[6]
        item.append(int(price1 == price2))
        item.append(math.fabs(price1 - price2))
        location1 = first[7]
        location2 = second[7]
        item.append(int(location1 == location2))
        metro1 = first[8]
        metro2 = second[8]
        if metro1 is None and metro2 is None:
            item.append(1) # notMetro attribute
            item.append(1) # metroEqual attr
        else:
            if metro1 is None or metro2 is None:
                item.append(1)
                item.append(0)
            else:
                item.append(0)
                if metro1 == metro2:
                    item.append(1)
                else:
                    item.append(0)

        lat1 = first[9]
        lat2 = second[9]
        item.append(int(lat1 == lat2))
        lon1 = first[10]
        lon2 = second[10]
        item.append(int(lon1 == lon2))
        item.append(math.fabs(lon1 - lon2))
        item.append(math.fabs(lat1 - lat2))

        insert_values.append(tuple(item))

    # with doesn't work
    cursor.executemany(insertQuery, insert_values)
    cursor.commit()
    cursor.close()
    print('insert_values were inserted')

def prepareData(data, shinglesLen, hash):
    pairs = data[0]
    X = []
    for pair in pairs:
        first = pair[0]
        second = pair[1]
        item = []
        cat1 = first[1]
        cat2 = second[1]
        if cat1 == cat2:
            item.append(1)
        else:
            item.append(0)
        if hash == 1:
            title1 = wordsToNumbers(textToVec(first[2]))
            title2 = wordsToNumbers(textToVec(second[2]))
        else:
            title1 = textToVec(first[2])
            title2 = textToVec(second[2])
        if shinglesLen == 0:
            titleSimilarity = computeJakkarSimilarity(title1,title2)
        else:
            titleSimilarity = computeShinglesJakkarSimilarity(title1,title2,shinglesLen)
        item.append(titleSimilarity)

        # Does the number of numbers in the titles match?
        numbers1 = [float(i) for i in title1 if (i.isdigit())]
        numbers2 = [float(i) for i in title2 if (i.isdigit())]
        if len(numbers1) == 0 and len(numbers2) == 0:
            item.append(1) # lengths equals
            item.append(1) # sim=1
            item.append(1) # median = 1
        elif len(numbers1) == 0 or len(numbers2) == 0:
            item.append(0)
            item.append(0)
            item.append(0)
        else:
            item.append(int(len(numbers1) == len(numbers2)))
            # the percentage of matching numbers
            title_numbers_similarity = compareNumbers(title1, title2)
            item.append(title_numbers_similarity)
            # median
            item.append(int(numpy.median(numbers1) == numpy.median(numbers2)))

        if hash == 1:
            description1 = wordsToNumbers(textToVec(first[3]))
            description2 = wordsToNumbers(textToVec(second[3]))
        else:
            description1 = textToVec(first[3])
            description2 = textToVec(second[3])
        if shinglesLen == 0:
            descriptionSimilarity = computeJakkarSimilarity(title1,title2)
        else:
            descriptionSimilarity = computeShinglesJakkarSimilarity(title1,title2,shinglesLen)
        item.append(descriptionSimilarity)

        # Does the number of numbers in the descriptions match?
        numbers1 = [float(i) for i in description1 if (i.isdigit())]
        numbers2 = [float(i) for i in description2 if (i.isdigit())]
        if len(numbers1) == 0 and len(numbers2) == 0:
            item.append(1) # lengths equals
            item.append(1) # sim=1
            item.append(1) # median = 1
        elif len(numbers1) == 0 or len(numbers2) == 0:
            item.append(0)
            item.append(0)
            item.append(0)
        else:
            # item.append(1)
            item.append(int(len(numbers1) == len(numbers2)))

            # the percentage of matching numbers
            title_numbers_similarity = compareNumbers(title1, title2)
            item.append(title_numbers_similarity)

            # median
            item.append(int(numpy.median(numbers1) == numpy.median(numbers2)))

        # compare title vs description and so on
        if shinglesLen == 0:
            title1Desc2 = computeJakkarSimilarity(title1, description2)
            title2Desc1 = computeJakkarSimilarity(title2, description1)
            item.append(title1Desc2)
            item.append(title2Desc1)
        else:
            title1Desc2 = computeShinglesJakkarSimilarity(title1, description2, shinglesLen)
            title2Desc1 = computeShinglesJakkarSimilarity(title2, description1, shinglesLen)
            item.append(title1Desc2)
            item.append(title2Desc1)

        title1.extend(description1)
        title2.extend(description2)
        if shinglesLen==0:
            titleDescSim = computeJakkarSimilarity(title1, title2)
        else:
            titleDescSim = computeShinglesJakkarSimilarity(title1, title2, shinglesLen)
        item.append(titleDescSim)

        # Does the number of numbers in the title + description match?
        numbers1 = [float(i) for i in title1 if (i.isdigit())]
        numbers2 = [float(i) for i in title2 if (i.isdigit())]

        if len(numbers1) == 0 and len(numbers2) == 0:
            item.append(1) # lengths equals
            item.append(1) # sim=1
            item.append(1) # median = 1
        elif len(numbers1) == 0 or len(numbers2) == 0:
            item.append(0)
            item.append(0)
            item.append(0)
        else:
            item.append(int(len(numbers1) == len(numbers2)))
            # the percentage of matching numbers
            title_numbers_similarity = compareNumbers(title1, title2)
            item.append(title_numbers_similarity)
            # median
            item.append(int(numpy.median(numbers1) == numpy.median(numbers2)))

        images_str1 = first[4]
        images_str2 = second[4]
        if images_str1 != '' and images_str2 != '':
            img_nums1 = images_str1.split(', ')
            img_nums2 = images_str2.split(', ')
            paths1 = create_paths(img_nums1)
            paths2 = create_paths(img_nums2)
            # Is number of images equals?
            if len(img_nums1) == len(img_nums2):
                item.append(1)
            else:
                item.append(0)
            computeHashes(item, paths1, paths2, 'ahash')
            computeHashes(item, paths1, paths2, 'phash')
            computeHashes(item, paths1, paths2, 'dhash')
            images1 = get_images(img_nums1)
            images2 = get_images(img_nums2)
            histograms1 = compute_histograms(images1, type='lab', hist_size=32)
            histograms2 = compute_histograms(images2, type='lab', hist_size=32)
            hist_f = compare_histograms(histograms1, histograms2, method="hellinger")
            item.append(hist_f)

        elif images_str1 != '' or images_str2 != '':
            item.append(0) # number of images doesn't equals
            # three types of hashes
            item.append(0)
            item.append(0)
            item.append(0)
            # not one image histograms are different
            item.append(0)
        else:
            # item.append(0) # there are not images
            # item.append(1) # to show that both ads don't have images
            item.append(1) # number of images equals and = 0
            # three types of hashes
            # 0 because not images
            # item.append(0)
            item.append(1)
            item.append(1) # to equate 0 and 1 in the hash and hist features
            item.append(1)
            item.append(1)

        if hash == 1:
                 attrsJSON1 = wordsToNumbers(textToVec(first[5]))
                 attrsJSON2 = wordsToNumbers(textToVec(second[5]))
        else:
            attrsJSON1 = textToVec(first[5])
            attrsJSON2 = textToVec(second[5])
        if shinglesLen == 0:
            attrsSim = computeJakkarSimilarity(attrsJSON1, attrsJSON2)
        else:
            attrsSim = computeShinglesJakkarSimilarity(title1,title2,shinglesLen)

        item.append(attrsSim)
        price1 = first[6]
        price2 = second[6]

        item.append(int(price1 == price2))
        if price1 == price2:
            item.append(0)
        else:
            item.append(1 / math.fabs(price1-price2))

        location1 = first[7]
        location2 = second[7]
        item.append(int(location1==location2))
        metro1 = first[8]
        metro2 = second[8]
        if metro1 is None and metro2 is None:
            item.append(1) # metroEqual attr
        else:
            if metro1 is None or metro2 is None:
                item.append(0)
            else:
                if metro1 == metro2:
                    item.append(1)
                else:
                    item.append(0)

        lat1 = first[9]
        lat2 = second[9]
        item.append(int(lat1==lat2))
        lon1 = first[10]
        lon2 = second[10]
        item.append(int(lon1==lon2))
        if lon1 == lon2:
            item.append(1) # lon closeness
        else:
            item.append(1 / (math.fabs(lon1-lon2)))

        if lat1 == lat2:
            item.append(1) # lat closeness
        else:
            item.append(1 / (math.fabs(lat1-lat2)))

        X.append(item)
    # X = preprocessing.normalize(X, norm='max', axis=0)
    Y = data[1]
    preparedData = (X,Y)
    return preparedData

def test_hashes(paths1, paths2):
    hashes_1 = []
    hashes_2 = []
    types = ["ahash", "phash", "dhash"]

    for type in types:
        for path1 in paths1:
            hashes_1.append(getHash(path1, type))

        for path2 in paths2:
            hashes_2.append(getHash(path2, type))

        hash_feature = 0

        for hash1 in hashes_1:
            for hash2 in hashes_2:
                hash_dist = hamming_dist(toBinStr(hash1), toBinStr(hash2))
                if hash_dist <= 10:
                    hash_feature = 1
        print("dist " + type, hash_feature )


def get_hashes(images, type):
    hashes = []
    for img in images:
        if type == 'ahash':
            hash = imagehash.average_hash(img)
        elif type == 'phash':
            hash = imagehash.phash(img)
        else:
            hash = imagehash.dhash(img)
        hashes.append(hash)
    return hashes

def computeHashes(item, paths1, paths2, type):
    hashes_1 = []
    hashes_2 = []

    for path1 in paths1:
        hashes_1.append(getHash(path1, type))

    for path2 in paths2:
        hashes_2.append(getHash(path2, type))

    hash_feature = 0

    for hash1 in hashes_1:
        for hash2 in hashes_2:
            hash_dist = hamming_dist(toBinStr(hash1), toBinStr(hash2))
            if(hash_dist <= 10):
                hash_feature = 1
                item.append(hash_feature)
                return

    item.append(hash_feature)

def compareAttrs(cursor, limit, whatCompare):
    if whatCompare == 'Duplicate':
        data = readAttrs(cursor, limit, 'Duplicate')
    else:
        data = readAttrs(cursor, limit, 'NoDuplicate')

    vec = prepareDuplicates(data)
    print(vec)
    res = checkDuplicates(vec)
    print(res)
    percent = computePercent(res)
    print('Right percent',percent)

def predictSVM(trainX, trainY, testX, testY):
    parameters = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],#, 0.8, 0.9, 1, 1.5, 2],
                  'kernel': ('linear', 'poly', 'sigmoid', 'rbf'),
          }
    clf = SVC()
    gscv = GridSearchCV(clf, parameters,'f1')
    gscv.fit(trainX,trainY)
    print("cv_results", gscv.cv_results_)
    print('best estimator',gscv.best_estimator_)
    print('best params',gscv.best_params_)
    print('best score',gscv.best_score_)

    predictRes = numpy.array(gscv.predict(testX))
    checkAnsw = []
    for i in range(len(testY)):
        checkAnsw.append(int(predictRes[i] == testY[i]))

    checkAnsw = numpy.array(checkAnsw)

    print('Percent of right answers', checkAnsw.sum()/len(checkAnsw))
    print('SVM precision', precision_score(testY,predictRes))
    print('SVM recall', recall_score(testY,predictRes))
    print('SVM f1', f1_score(testY,predictRes))


def predict_svm_test_time(trainX, trainY, testX, testY):
    clf = SVC(C=0.7, kernel='sigmoid')
    clf.fit(trainX, trainY)
    start_time = time.time()
    numpy.array(clf.predict(testX))
    end_time = time.time()
    return end_time - start_time

def predict_dt_test_time(trainX, trainY, testX, testY):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_leaf=1)
    clf.fit(trainX, trainY)
    start_time = time.time()
    numpy.array(clf.predict(testX))
    end_time = time.time()
    return end_time - start_time

def predict_rf_test_time(trainX, trainY, testX, testY):
    clf = RandomForestClassifier(n_estimators=23, max_depth=19, min_samples_leaf=7)
    clf.fit(trainX, trainY)
    start_time = time.time()
    numpy.array(clf.predict(testX))
    end_time = time.time()
    return end_time - start_time

def predictDT(trainX, trainY, testX, testY):
    parameters = {
        'criterion':['gini','entropy'],
        'max_depth': [i for i in range(1, 10)],
        'min_samples_leaf':[i for i in range(1,15)]
        }
    clf = DecisionTreeClassifier()
    gscv = GridSearchCV(clf,parameters,'f1')
    gscv.fit(trainX,trainY)
    print('best estimator',gscv.best_estimator_)
    print('best params',gscv.best_params_)
    print('best score',gscv.best_score_)

    # probability
    # proba =  numpy.array(gscv.predict_proba(testX))
    # print('DT AUC', roc_auc_score(testY, proba[:,1]))

    predictRes = numpy.array(gscv.predict(testX))
    checkAnsw = []
    for i in range(len(testY)):
        checkAnsw.append(int(predictRes[i] == testY[i]))

    checkAnsw = numpy.array(checkAnsw)

    print('DT score', checkAnsw.sum()/len(checkAnsw))
    print('DT precision', precision_score(testY,predictRes))
    print('DT recall', recall_score(testY,predictRes))
    print('DT f1', f1_score(testY,predictRes))

def predict_knn_test_time(trainX, trainY, testX, testY):
    clf = KNeighborsClassifier(n_neighbors=43, metric='manhattan')
    clf.fit(trainX, trainY)
    start_time = time.time()
    numpy.array(clf.predict(testX))
    end_time = time.time()
    return end_time - start_time

def predictLogReg(_trainX, _trainY, _textX, _testY):
    model = LogisticRegression()
    # Create regularization penalty space
    penalty = ['l1', 'l2']

    # Create regularization hyperparameter space
    C = numpy.logspace(0, 4, 10)

    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty)
    # Create grid search using 5-fold cross validation
    clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)
    best_model = clf.fit(_trainX, _trainY)
    predictRes = best_model.predict(_textX)
    print('LogReg precision', precision_score(_testY,predictRes))
    print('LogReg recall', recall_score(_testY,predictRes))
    print('LogReg f1', f1_score(_testY,predictRes))

def predictRandomForest(_trainX, _trainY, _textX, _testY):
    clf = RandomForestClassifier()
    parameters = {'n_estimators': [i for i in range(23, 24)],
                  'max_features':[i for i in range(23, 26)],
                  # 'criterion':["gini", "entropy"],
                  'max_depth': [i for i in range(19, 20)],
                  'min_samples_leaf':[i for i in range(7,8)],
                  'n_jobs': [-1]
                  }
    clf = GridSearchCV(clf,parameters,'f1')
    clf.fit(_trainX, _trainY)
    predictRes = clf.predict(_textX)
    print('best estimator', clf.best_estimator_)
    print('best params', clf.best_params_)
    print('best score', clf.best_score_)
    print('RandomForest precision', precision_score(_testY,predictRes))
    print('RandomForest recall', recall_score(_testY,predictRes))
    print('RandomForest f1', f1_score(_testY,predictRes))


def predictKNN(trainX, trainY, testX, testY):
    parameters = {'n_neighbors': [i for i in range(43, 50)],
                  'weights':['distance', 'uniform'],
                  'metric':['euclidean', 'manhattan']}

    clf = KNeighborsClassifier()

    gscv = GridSearchCV(clf,parameters,'f1')
    gscv.fit(trainX,trainY)
    print("cv results", gscv.cv_results_)
    print('best estimator',gscv.best_estimator_)
    print('best params',gscv.best_params_)
    print('best score',gscv.best_score_)

    # # probability
    # predict_proba return two probabilities
    # proba = gscv.predict_proba(testX)
    # predictRes = numpy.array(gscv.predict_proba(testX))
    # print('KNN AUC', roc_auc_score(testY, predictRes[:,1]))

    predictRes = gscv.predict(testX)
    checkAnsw = []
    for i in range(len(testY)):
        checkAnsw.append(int(predictRes[i] == testY[i]))

    checkAnsw = numpy.array(checkAnsw)

    print('KNN score', checkAnsw.sum()/len(checkAnsw))
    print('KNN precision', precision_score(testY,predictRes))
    print('KNN recall', recall_score(testY,predictRes))
    print('KNN f1', f1_score(testY,predictRes))


def wordsToNumbers(words):
    numbers = [binascii.crc32(word.encode()) for word in words]
    return numbers

def hamming_dist(str1, str2):
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs

def hamming_similarity(str1, str2):
    sim = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 == ch2:
            sim += 1
    return sim

def toBinStr(h):
    h = str(h)
    h_size = len(h) * 4
    return (bin(int(h, 16))[2:]).zfill(h_size)

def getHash(path, type='phash'):
    img = Image.open(path)
    if type == 'ahash':
        hash = imagehash.average_hash(img)
    elif type == 'phash':
        hash = imagehash.phash(img)
    else:
        hash = imagehash.dhash(img)

    return hash

def imageHash():
    path = 'E:\\Study\\Diploma\\Avito duplicates\\'
    num1 = '9458537'
    num2 = '13325040'
    last2 = num1[-2:]
    if last2[0] == '0':
        path1 = path + 'Images_' + last2[0] + '\\' + last2[1] + '\\' + num1 + ".jpg"
    else:
        path1 = path + 'Images_' + last2[0] + '\\' + last2 + '\\' + num1 + ".jpg"

    last2 = num2[-2:]
    if last2[0] == '0':
        path2 = path + 'Images_' + last2[0] + '\\' + last2[1] + '\\' + num2 + ".jpg"
    else:
        path2 = path + 'Images_' + last2[0] + '\\' + last2 + '\\' + num2 + ".jpg"

    img1 = Image.open(path1)
    img1.format = "JPG"
    img1.show()
    img2 = Image.open(path2)
    img2.format = "JPG"
    img2.show()
    hash1 = imagehash.average_hash(img1)
    hash2 = imagehash.average_hash(img2)
    print('average hash1',hash1)
    print('average hash2',hash2)
    print('ahash dist', hamming_dist(toBinStr(hash1), toBinStr(hash2)))

    hash1 = imagehash.phash(Image.open(path1))
    hash2 = imagehash.phash(Image.open(path2))
    print('p hash1',hash1)
    print('p hash2',hash2)
    print('phash dist', hamming_dist(toBinStr(hash1), toBinStr(hash2)))

    hash1 = imagehash.dhash(Image.open(path1))
    hash2 = imagehash.dhash(Image.open(path2))
    print('d hash1',hash1)
    print('d hash2',hash2)
    print('dhash dist', hamming_dist(toBinStr(hash1), toBinStr(hash2)))

def siftTest():
    path = 'E:/Study/Diploma/Avito duplicates/'
    num1 = '9458537'
    num2 = '13325040'
    last2 = num1[-2:]
    if last2[0] == '0':
        path1 = path + 'Images_' + last2[0] + '/' + last2[1] + '/' + num1 + ".jpg"
    else:
        path1 = path + 'Images_' + last2[0] + '/' + last2 + '/' + num1 + ".jpg"

    last2 = num2[-2:]
    if last2[0] == '0':
        path2 = path + 'Images_' + last2[0] + '\\' + last2[1] + '\\' + num2 + ".jpg"
    else:
        path2 = path + 'Images_' + last2[0] + '\\' + last2 + '\\' + num2 + ".jpg"

    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    #print('img info', img.shape)
    gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)
    print('matches', matches)
    dist = [m.distance for m in matches]

    print('distance: min: %.3f' % min(dist))
    print('distance: mean: %.3f' % (sum(dist) / len(dist)))
    print('distance: max: %.3f' % max(dist))

    # threshold: half the mean
    thres_dist = (sum(dist) / len(dist)) * 0.5

    # keep only the reasonable matches
    sel_matches = [m for m in matches if m.distance < thres_dist]
    print('len sel_matches', len(sel_matches))
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
    view[:h1, :w1, :] = img1
    view[:h2, w1:, :] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]

    for m in sel_matches:
        # draw the keypoints
        color = tuple([sp.random.randint(0, 255) for _ in range(3)])
        cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])) , (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), color)


    cv2.imshow("view", view)
    cv2.waitKey()

def mserTest():
    path = 'E:/Study/Diploma/Avito duplicates/'
    num1 = '1064094'
    num2 = '1227519'
    last2 = num1[-2:]
    if last2[0] == '0':
        path1 = path + 'Images_' + last2[0] + '/' + last2[1] + '/' + num1 + ".jpg"
    else:
        path1 = path + 'Images_' + last2[0] + '/' + last2 + '/' + num1 + ".jpg"

    last2 = num2[-2:]
    if last2[0] == '0':
        path2 = path + 'Images_' + last2[0] + '\\' + last2[1] + '\\' + num2 + ".jpg"
    else:
        path2 = path + 'Images_' + last2[0] + '\\' + last2 + '\\' + num2 + ".jpg"

    img1 = cv2.imread(path1,0) # 0 - grayscale
    img2 = cv2.imread(path2,0) # 0 - grayscale
    mser = cv2.MSER_create()
    vis1 = img1.copy()

    regions, _ = mser.detectRegions(img1)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis1, hulls, 1, (0, 255, 0))

    vis2 = img2.copy()
    regions, _ = mser.detectRegions(img2)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis2, hulls, 1, (0, 255, 0))
    fig = plt.figure()
    images = []
    images.append(vis1)
    images.append(vis2)
    for (i,image) in enumerate(images):
        ax = fig.add_subplot(1, len(images), i + 1)
        plt.axis('off')
        plt.imshow(image)

    plt.draw()
    plt.show()


def orbTest():
    path = 'E:/Study/Diploma/Avito duplicates/'
    num1 = '1064094'
    num2 = '1227519'
    last2 = num1[-2:]
    if last2[0] == '0':
        path1 = path + 'Images_' + last2[0] + '/' + last2[1] + '/' + num1 + ".jpg"
    else:
        path1 = path + 'Images_' + last2[0] + '/' + last2 + '/' + num1 + ".jpg"

    last2 = num2[-2:]
    if last2[0] == '0':
        path2 = path + 'Images_' + last2[0] + '\\' + last2[1] + '\\' + num2 + ".jpg"
    else:
        path2 = path + 'Images_' + last2[0] + '\\' + last2 + '\\' + num2 + ".jpg"

    img1 = cv2.imread(path1,0) # 0 - grayscale
    img2 = cv2.imread(path2,0) # 0 - grayscale
    orb = cv2.ORB_create()
    kp = orb.detect(img1, None)
    kp1, des1 = orb.compute(img1, kp)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    dist = [m.distance for m in matches]

    print('distance: min: %.3f' % min(dist))
    print('distance: mean: %.3f' % (sum(dist) / len(dist)))
    print('distance: max: %.3f' % max(dist))

    mean = sum(dist) / len(dist)
    # keep only the reasonable matches
    sel_matches = [m for m in matches if m.distance < mean]
    print('len sel_matches', len(sel_matches))
    # Sort them in the order of their distance.
    sel_matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2, sel_matches[:5],None, flags=2) # bad find matching places
    plt.imshow(img3),plt.show()


def get_cursor():
    server = 'DMITRIY\SQLSERVER2017'
    database = 'Avito'
    username = 'test_user'
    password = '123'
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 13 for SQL Server};SERVER=' + server + ';PORT=1443;DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    cursor = conn.cursor()

    return (cursor,conn)

def create_paths(nums):
    paths = []
    path = 'E:\\Study\\Diploma\\Avito duplicates\\'
    for num in nums:
        last2 = num[-2:]
        if last2[0] == '0':
            path1 = path + 'Images_' + last2[0] + '\\' + last2[1] + '\\' + num + ".jpg"
        else:
            path1 = path + 'Images_' + last2[0] + '\\' + last2 + '\\' + num + ".jpg"

        paths.append(path1)

    return paths

def get_images(nums1):
    paths = create_paths(nums1)
    images = []
    for path in paths:
        image = cv2.imread(path)
        images.append(image)
    return images

def compute_histograms(images, type="rgb", hist_size=32):
    histograms = []
    for image in images:
        if type == "rgb":
            cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB, dst=image)
            # image = cv2.resize(image, (100, 100))
            hist = cv2.calcHist([image], [0, 1, 2], None, [hist_size, hist_size, hist_size], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten() # flatten - make one-dimensional list
            histograms.append(hist)
        else:
            cv2.cvtColor(image, cv2.COLOR_BGR2Lab, image)
            hist = cv2.calcHist([image], [1, 2], None, [hist_size, hist_size], [0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten() # flatten - make one-dimensional list
            histograms.append(hist)
    return histograms

def compare_hashes(images_num1, images_num2, type):
    paths1 = create_paths(images_num1)
    paths2 = create_paths(images_num2)
    hashes_1 = []
    hashes_2 = []

    for path1 in paths1:
        hashes_1.append(getHash(path1, type))

    for path2 in paths2:
        hashes_2.append(getHash(path2, type))

    hash_feature = 0

    for hash1 in hashes_1:
        for hash2 in hashes_2:
            hash_dist = hamming_dist(toBinStr(hash1), toBinStr(hash2))
            if hash_dist <= 10:
                img_num1 = images_num1[hashes_1.index(hash1)]
                img_num2 = images_num2[hashes_2.index(hash2)]
                print("Images with similar hashes", img_num1, img_num2)
                hash_feature = 1
                return hash_feature

    return hash_feature

def compare_histograms(histograms1, histograms2, method='correlation'):
    feature = 0
    if method == "correlation":
        _method = cv2.HISTCMP_CORREL
    else:
        _method = cv2.HISTCMP_BHATTACHARYYA
    for hist1 in histograms1:
        for hist2 in histograms2:
                dist = cv2.compareHist(hist1, hist2, _method)
                if _method == cv2.HISTCMP_CORREL:
                    if dist > 0.5:
                        feature = 1
                        return feature
                else:
                    if dist < 0.5:
                        feature = 1
                        return feature

    return feature

def compare_histograms_new(images_num1, images_num2):
    images1 = get_images(images_num1)
    images2 = get_images(images_num2)
    histograms1 = compute_histograms(images1, "lab")
    histograms2 = compute_histograms(images2, "lab")
    feature = 0
    _method = cv2.HISTCMP_BHATTACHARYYA
    i = 0
    j = 0
    for hist1 in histograms1:
        for hist2 in histograms2:
            dist = cv2.compareHist(hist1, hist2, _method)
            if _method == cv2.HISTCMP_CORREL:
                if dist > 0.5:
                    feature = 1
                    return feature
            else:
                if dist < 0.5:
                    print("Images with similar histograms", images_num1[i], images_num2[j])
                    feature = 1
                    return feature
            j+=1
        i+=1
    return feature

def compute_histogram_distances_minimum(histograms1, histograms2, method):
    distances = []
    if method == "correlation":
        _method = cv2.HISTCMP_CORREL
    else:
        _method = cv2.HISTCMP_BHATTACHARYYA

    for hist1 in histograms1:
        for hist2 in histograms2:
            dist = cv2.compareHist(hist1, hist2, _method)
            distances.append(dist)

    return numpy.array(distances).min()

def compute_histograms_threshold(train_data, method):
    pairs = train_data[0]
    h_distances_min = []
    for pair in pairs:
        first = pair[0]
        second = pair[1]
        images_str1 = first[4]
        images_str2 = second[4]
        if images_str1 != '' and images_str2 != '':
            img_nums1 = images_str1.split(', ')
            img_nums2 = images_str2.split(', ')
            images1 = get_images(img_nums1)
            images2 = get_images(img_nums2)
            histograms1 = compute_histograms(images1, type='lab', hist_size=32)
            histograms2 = compute_histograms(images2, type='lab', hist_size=32)
            distances_min = compute_histogram_distances_minimum(histograms1=histograms1, histograms2=histograms2, method=method)
            h_distances_min.append(distances_min)

    return numpy.array(h_distances_min).max()

def show_images(images_num):
    images = get_images(images_num)
    fig = plt.figure()
    for (i,image) in enumerate(images):
        cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB, dst=image)
        fig.add_subplot(1, len(images), i + 1)
        plt.axis('off')
        image_num = images_num[i]
        plt.title(str(image_num))
        plt.imshow(image)

    plt.draw()

def test_histograms(nums=None, hist_size=32, space="rgb"):

    if nums is None:
        nums = []
    imageWorker = ImageWorker()
    imageWorker.set_images_nums(nums)

    if space == "rgb":
        # RGB
        imageWorker.read_color_images()
        imageWorker.show_images()
        hist = imageWorker.compute_histograms(space="rgb", hist_size = hist_size)
        imageWorker.compare_histograms(histograms=hist)
        imageWorker.show_rgb_histograms()
    else:
        # Lab
        imageWorker.read_lab_images()
        imageWorker.show_images()
        hist = imageWorker.compute_histograms(space="lab", hist_size = hist_size)
        imageWorker.compare_histograms(histograms=hist)
        imageWorker.show_lab_histograms(histograms=hist)

    plt.show()

def test_to_watch_images():
    nums1_str = '1458586, 1966380, 5224720, 5391973'
    nums2_str = '10410282, 11554289, 3681463, 4654953, 6730701'

    nums1 = nums1_str.split(", ")
    nums2 = nums2_str.split(", ")

    images_1 = get_images(nums1)
    images_2 = get_images(nums2)

    show_images(images_1)
    show_images(images_2)

    plt.show()

def compare_ads(cursor, trainX, trainY, testX, testY, method, limit=9000,  pair_number=1):
    selectQuery = 'select top ' + str(limit) + ' * from MixedData '
    Y = []
    X = []
    with cursor.execute(selectQuery):
        row = cursor.fetchone()
        index = 1
        while row:
            if index > 6000:
                Y.append(row[1])
                first = row[2:13]
                second = row[13:]
                X.append((first,second))
            row = cursor.fetchone()
            index += 1

    if method == "dt":
        clf = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=10)
    elif method == "svm":
        clf = SVC(C=0.5, kernel='sigmoid')
    elif method == "knn":
        clf = KNeighborsClassifier(n_neighbors=45, metric='manhattan')
    else:
        clf = RandomForestClassifier(n_estimators=33, criterion='gini', max_depth=20, max_features=23, min_samples_leaf=4)
    clf.fit(trainX,trainY)
    testX = numpy.reshape(testX[pair_number-1], (1,-1))
    predictRes = numpy.array(clf.predict(testX))
    print("Predicted result", predictRes)
    index = 0
    for pair in X:
        if index == pair_number - 1:

            first = pair[0]
            second = pair[1]
            print("Duplicate ", Y[index])
            print("Первое объявление с id", first[0])
            print("categoryID: ", first[1])
            print("title: ", first[2])
            print("description: ", first[3])
            print("images_array: ", first[4])
            print("attrsJSON: ", first[5])
            print("price: ", first[6])
            print("locationID: ", first[7])
            print("metroID: ", first[8])
            print("lat: ", first[9])
            print("lon: ", first[10])

            print("//////////////////////////////////////////////////////////////////////////////////////////////////")

            print("Второе объявление с id", second[0])
            print("categoryID: ", second[1])
            print("title: ", second[2])
            print("description: ", second[3])
            print("images_array: ", second[4])
            print("attrsJSON: ", second[5])
            print("price: ", second[6])
            print("locationID: ", second[7])
            print("metroID: ", second[8])
            print("lat: ", second[9])
            print("lon: ", second[10])
            print("-------------------------------------------------------------------------------------------------")
            title1 = first[2]
            title2 = second[2]
            print("Title similarity", computeJakkarSimilarity(textToVec(title1), textToVec(title2)))
            desc1 = first[3]
            desc2 = second[3]
            print("Descriptions similarity", computeJakkarSimilarity(textToVec(desc1), textToVec(desc2)))
            images_num1 = first[4].split(", ")
            images_num2 = second[4].split(", ")
            show_images(images_num1)
            show_images(images_num2)
            price1 = first[6]
            price2 = second[6]
            print("Price similarity", int(price1 == price2))
            loc1 = first[7]
            loc2 = second[7]
            print("Location similarity", int(loc1 == loc2))
            ahash_result = compare_hashes(images_num1, images_num2, 'ahash')
            print("Ahash similar", ahash_result)
            phash_result = compare_hashes(images_num1, images_num2, 'phash')
            print("Phash similar", phash_result)
            dhash_result = compare_hashes(images_num1, images_num2, 'dhash')
            print("Dhash similar", dhash_result)
            hist_similarity = compare_histograms_new(images_num1, images_num2)
            print("Histograms similarity", hist_similarity)
            plt.show()
            print("-------------------------------------------------------------------------------------------------")
            break
        index += 1

def feature_selection(X,Y):
    # Feature selection
    # All features have high rating
    model = SVC(kernel='linear')
    selector = RFECV(estimator=model, step=1, scoring='f1')
    selector = selector.fit(X, Y)
    print("Optimal number of features : %d" % selector.n_features_)
    print("Selected Features: ", selector.support_)
    print("Feature Ranking: ", selector.ranking_)

    # model = ExtraTreesClassifier()
    # model.fit(X, Y)
    # print(model.feature_importances_)

if __name__ == '__main__':

    (cursor, conn) = get_cursor()

    # test of prediction

    dataSize = 9000
    read_text_features = False
    if read_text_features:
        data = readTextFeatures(cursor, dataSize)
    else:
        data = readNewFeatures(cursor, dataSize)

    X = data[0]
    Y = data[1]
    trainCount = int(2 * dataSize / 3)
    testCount = int(dataSize/3)
    trainX = numpy.array(X[:trainCount])
    trainY = numpy.array(Y[:trainCount])
    testX = numpy.array(X[trainCount:])
    testY = numpy.array(Y[trainCount:])

    # # Compare advertisements for attachment
    # compare_ads(cursor, trainX, trainY, testX, testY, method="rf", pair_number=33)

    method = "rf"
    test_time = False
    runsCount = 100
    elapsed_time = 0
    if test_time:
        for i in range(0, runsCount):
            if method == "knn":
                elapsed_time += predict_knn_test_time(trainX, trainY, testX, testY)
            elif method == "svm":
                elapsed_time += predict_svm_test_time(trainX, trainY, testX, testY)
            elif method == "dt":
                elapsed_time += predict_dt_test_time(trainX, trainY, testX, testY)
            elif method == "rf":
                elapsed_time += predict_rf_test_time(trainX, trainY, testX, testY)
    else:
        if method == "knn":
            predictKNN(trainX, trainY, testX, testY)
        elif method == "svm":
            predictSVM(trainX, trainY, testX, testY)
        elif method == "dt":
            predictDT(trainX, trainY, testX, testY)
        elif method == "rf":
            predictRandomForest(trainX, trainY, testX, testY)

    if test_time:
        print("Mean elapsed time", elapsed_time / runsCount)

    conn.commit()
    conn.close()
