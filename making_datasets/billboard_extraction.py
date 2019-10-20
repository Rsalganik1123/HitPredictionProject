import re 
import billboard
from datetime import datetime, date, timedelta
import time
import random 

class Song:
    # songCount = 0
    # songDictionary = {}

    def __init__(self, title):
        
        self.artistName = None
        self.title = title
        self.year = None
def populate(chart): 
    outputFile = open('BillboardCSV.csv', 'w')
    # csvRowString = ""
    # print(chart)
    for s in chart: 
        csvRowString = ""
        song = Song(s.title)
        song.artistName = s.artist
        csvRowString += song.title + "," + song.artistName + "\n"
        outputFile.write(csvRowString)
        
    outputFile.close()

def main(): 
    fmt = "%Y-%m-%d"
    
    start_time = datetime.strptime(billboard.ChartData('hot-100').date, fmt)
    end_time = datetime(year=1990, month=1, day=1)
    week = timedelta(days=7)
    # offset = random.randint(1, 29)
    

    # outputFile = open('BillboardCSV.csv', 'a+')
    csvRowString = "Title,ArtistName" + "\n"
    # outputFile.write(csvRowString)
    with open('../Datasets/BillboardCSV1990.csv', 'a') as outputFile: 
        outputFile.write(csvRowString)
        for i in range(1000): 
            offset = random.randint(1, 520) #1508 for 1990, 468 for 2010 , 260 for 2014
            date = start_time - week*offset
            date_str = datetime.strftime(date, fmt)
            print(date_str)
            chart = billboard.ChartData('hot-100', date = date_str)
            print(chart[:5])
            for s in chart: 
                csvRowString = ""
                song = Song(s.title.replace(",", " ").lower())
                song.artistName = s.artist.lower()
                csvRowString += song.title + "," + song.artistName + "\n"
                outputFile.write(csvRowString)
            time.sleep(10)
    
    
    # chart = billboard.ChartData('hot-100')
    # while chart.previousDate:
    #     date_str= chsart.date 
    #     print(date_str) 
    #     chart = billboard.ChartData('hot-100', date=date_str)
    #     date = datetime.strptime(chart.date, fmt)
    #     if date > end_time: 
            
    #         # populate(chart)
    #         for s in chart: 
    #             csvRowString = ""
    #             song = Song(s.title)
    #             song.artistName = s.artist
    #             csvRowString += song.title + "," + song.artistName + "\n"
    #             outputFile.write(csvRowString)
    #     else:    
    #         break
    #     chart = billboard.ChartData('hot-100', date = chart.previousDate)
    #     time.sleep(10)
    # outputFile.close() 

# def test():
    # chart = billboard.ChartData('hot-100')
    # ChartData(name, date=None, fetch=True, timeout=25)
    # print(chart)
    
main()
# test()