import numpy as np
import json
from object_helper import ObjectHelper
from list_helper import ListHelper
from json_helper import JSONHelper
from file_helper import FileHelper
from input import inputs
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,BatchNormalization,LSTM
from keras.optimizers import SGD,Adam
from keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split


dataPath = './data.json'
def loadData():
    with open(dataPath, 'r') as f:
        data = json.load(f)
    return data

def countRepeat(listNum):
    return ListHelper.reduce(
        listNum,
        func= lambda acc, current : ObjectHelper.assign(
        acc, 
            {current: acc[current] + 1} if ObjectHelper.has_key(acc, current) else {current: 1}
    ),
        initial={}
    )

def sortRepeated(repeated):
    repeatedInList = ObjectHelper.entries(repeated)
    repeatedInList = ListHelper.sort(repeatedInList, lambda a, b: a[1] - b[1])
    return ListHelper.map(repeatedInList, lambda x: x[0])

def getNum(data):
    listNum01 = []
    listNum02 = []
    listNum03 = []
    listNum04 = []
    listNum05 = []
    listNum06 = []
    for item in data:
        listNum01.append(item['num_01'])
        listNum02.append(item['num_02'])
        listNum03.append(item['num_03'])
        listNum04.append(item['num_04'])
        listNum05.append(item['num_05'])
        listNum06.append(item['num_06']) 
    return listNum01, listNum02, listNum03, listNum04, listNum05, listNum06

def sortRepeatedNum(listNum01,listNum02,listNum03,listNum04,listNum05,listNum06):
    return {
        "num_01": sortRepeated(countRepeat(listNum01)),
        "num_02": sortRepeated(countRepeat(listNum02)),
        "num_03": sortRepeated(countRepeat(listNum03)),
        "num_04": sortRepeated(countRepeat(listNum04)),
        "num_05": sortRepeated(countRepeat(listNum05)),
        "num_06": sortRepeated(countRepeat(listNum06)),
    }

def convertRawDataToIndex(data,sortedNum):
    return ListHelper.map(
        data,
        lambda x: [
            ListHelper.find_index(sortedNum["num_01"], lambda y: y == x["num_01"]),
            ListHelper.find_index(sortedNum["num_02"], lambda y: y == x["num_02"]),
            ListHelper.find_index(sortedNum["num_03"], lambda y: y == x["num_03"]),
            ListHelper.find_index(sortedNum["num_04"], lambda y: y == x["num_04"]),
            ListHelper.find_index(sortedNum["num_05"], lambda y: y == x["num_05"]),
            ListHelper.find_index(sortedNum["num_06"], lambda y: y == x["num_06"])
        ]
    )

def extractData(data,fromIndex,toIndex):
    return ListHelper.map(
        data,
        lambda x: x[fromIndex:toIndex+1]
    )

def processInputOutput(data,numOfSet):

    input = []
    output = []

    for i in range((len(data) - numOfSet)+1):
        try:
            inputBatch = data[i:i+numOfSet]
            outputBatch = data[i+numOfSet]
            input.append(inputBatch)
            output.append(outputBatch)
        except IndexError as err:
            break    

    return input, output

def buildModel(inputShape,numOfFeature):

    model = Sequential([
        Dense(16,activation='tanh', input_shape=inputShape),
        Dense(numOfFeature,activation='relu'),
    ])

    model.compile(
        optimizer=SGD(),
        loss='mse',
        metrics=['accuracy'],
    )

    return model

numOfSet = 8

def main():
    
    data = loadData()

    listNum01, listNum02, listNum03, listNum04, listNum05, listNum06 = getNum(data)

    sortedRepeatedNumDict = sortRepeatedNum(listNum01,listNum02,listNum03,listNum04,listNum05,listNum06)

    # JSONHelper.write_json(sortedRepeatedNumDict,'./sortedRepeatedNumDict.json')

    if FileHelper.file_exists('model.h5'):
        model = load_model('model.h5')
    else:
        
        rawDataToIndex = convertRawDataToIndex(data,sortedRepeatedNumDict)

        extractedData = extractData(rawDataToIndex,0,1)

        input,output = processInputOutput(extractedData,numOfSet)

        input = np.array(input)

        output = np.array(output)

        X_train,X_test, y_train,y_test = train_test_split(
            input,
            output,
            test_size=0.2,
            random_state=42
        )

        model = buildModel(
            (numOfSet,2),
            2
        )

        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=2,
            validation_data=(X_test,y_test)
        )

        #model.save('model.h5')

    input_data = inputs[:numOfSet]

    rawDataToIndex = convertRawDataToIndex(input_data,sortedRepeatedNumDict)
        
    extractedData = extractData(rawDataToIndex,0,1)

    input_data = np.array(extractedData)

    input_data = np.reshape(input_data,(-1,numOfSet,2))

    predicted = model.predict(input_data)

    predicted = np.round(predicted).astype(int)

    num_01 = predicted[0][0][0]

    num_02 = predicted[0][0][1]

    num_01 = sortedRepeatedNumDict["num_01"][num_01]

    num_02 = sortedRepeatedNumDict["num_02"][num_02]

    print(num_01,num_02)

if __name__ == '__main__':
    main()