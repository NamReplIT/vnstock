import numpy as np
import json
from object_helper import ObjectHelper
from list_helper import ListHelper
from json_helper import JSONHelper
from file_helper import FileHelper
from date_helper import DateHelper
from vnstock.fundamental import *
from vnstock.technical import *


def getListSymbol():
    listSymbol = []
    if FileHelper.file_exists('./json/stock/listSymbol.json',create_dir=True):
        listSymbol = JSONHelper.read_json('./json/stock/listSymbol.json')
    else:
        listSymbol = listing_companies(live=True).to_dict('records')
        JSONHelper.write_json('./json/stock/listSymbol.json', listSymbol)
        
    return ListHelper.map(listSymbol, lambda x: x['ticker'])

def getIntradayData(listSymbol):
    for symbol in listSymbol:
        intradayData = stock_historical_data(
            symbol = symbol,
            start_date = '2023-05-01',
            end_date = '2023-12-01',
            resolution = '1',
        ).to_dict('records')

        if FileHelper.file_exists(f"./json/intraday/{symbol}.json",create_dir=True) is False:
            JSONHelper.write_json(f"./json/intraday/{symbol}.json", intradayData)

def getHourlyData(listSymbol):
    for symbol in listSymbol:
        result = stock_historical_data(
            symbol = symbol,
            start_date = '2023-05-01',
            end_date = '2023-12-01',
            resolution = '1H',
        ).to_dict('records')

        if FileHelper.file_exists(f"./json/intraDaily/{symbol}.json",create_dir=True) is False:
            JSONHelper.write_json(f"./json/intraDaily/{symbol}.json", result)

def getDailyData(listSymbol):
    for symbol in listSymbol:
        result = stock_historical_data(
            symbol = symbol,
            start_date = '2023-05-01',
            end_date = '2023-12-01',
            resolution = '1D',
        ).to_dict('records')

        if FileHelper.file_exists(f"./json/intraDaily/{symbol}.json",create_dir=True) is False:
            JSONHelper.write_json(f"./json/intraDaily/{symbol}.json", result)


def getListOfIntraday():
    listFileName = FileHelper.read_dir('./json/intraday')
    return listFileName

def getListOfMinute():
    morningStart = DateHelper.parse_date('2000-01-01 09:00:00')
    afternoonStart = DateHelper.parse_date('2000-01-01 13:00:00')
    listMinute = []
    for i in range(0, 151):
        listMinute.append(DateHelper.format(
            DateHelper.addSeconds(morningStart, i * 60),
            DateHelper.TIME_24_HOUR_FORMAT
        ))
    for i in range(0, 121):
        listMinute.append(DateHelper.format(
            DateHelper.addSeconds(afternoonStart, i * 60),
            DateHelper.TIME_24_HOUR_FORMAT
        ))
    return listMinute

def getListOfWeekday():
    start = '2023-09-05'
    numOfDay = 54
    listWeekday = []
    for i in range(0, numOfDay):
        listWeekday.append(DateHelper.format(
            DateHelper.addDays(start, i,['Saturday','Sunday']),
            DateHelper.ISO_8601_DATE
        ))
    return listWeekday

def getMinuteTrading(date = '2023-09-05',symbol = 'VIC'):
    intradayData = JSONHelper.read_json(f"./json/intraday/{symbol}.json")
    intradayData = ListHelper.map(
        intradayData, 
        lambda x: ObjectHelper.assign(
            x,
            {"time": DateHelper.format(DateHelper.parse_date(x["time"]),f"{DateHelper.ISO_8601_DATE} %H:%M:00")}
        )
    )
    
    listMinute = getListOfMinute()

    listMinuteTrading = []
    for minute in listMinute:
        dateTime = f"{date} {minute}"
        minuteTrading = ListHelper.find(
            intradayData,
            lambda x: x["time"] == dateTime
        )
        if minuteTrading is not None:
            listMinuteTrading.append([
                minuteTrading["open"],
                minuteTrading["high"],
                minuteTrading["low"],
                minuteTrading["close"],
                minuteTrading["volume"],
            ])
        else:
            listMinuteTrading.append([0,0,0,0,0])
    return listMinuteTrading
    
def main():
    listSymbol = getListSymbol()
    getDailyData(listSymbol)
        

    

    

    
if __name__ == '__main__':
    main()