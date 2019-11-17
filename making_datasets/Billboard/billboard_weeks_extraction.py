import re 
import billboard
from datetime import datetime, date, timedelta
import time
import random 
import csv

class Song:
    # songCount = 0
    # songDictionary = {}

    def __init__(self, title):
        
        self.artistName = None
        self.title = title
        self.date = None
        self.rank = None 
        self.weeks = None
        self.new = None

def main(): 
    first = False
    with open('./Datasets/Billboard/ContinuousChart.csv', 'a+') as outputFile: 
        writer = csv.writer(outputFile, delimiter=',', lineterminator = '\n')
        if first: writer.writerow(["Date","Title","ArtistName","Rank","Weeks","isNew"])
        chart = billboard.ChartData('hot-100', date ='2017-12-30') 
        while '2016' not in chart.previousDate: 
            chart = billboard.ChartData('hot-100', date = chart.previousDate)
            print(chart.previousDate, '\n', chart[:5])
            for i in range(len(chart)): 
                s = chart[i]
                
                song = Song(s.title.replace(",", " ").lower())
                if s.isNew:
                    song.new = '1' 
                else: song.new = '0' 
                song.artistName = s.artist.replace(",", " ").lower()
                song.rank = s.rank
                song.weeks = s.weeks
                song.date = chart.previousDate  
                writer.writerow([song.date, song.title,song.artistName, song.rank, song.weeks, song.new ])
                
            time.sleep(10)

def test():
    print('here')
    #chart = billboard.ChartData('hot-100')
    chart = billboard.ChartData('hot-100', date='2019-10-26') 
    print(chart)
       
main()
#test()