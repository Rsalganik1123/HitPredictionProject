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
        self.year = None
        self.rank = None 
        self.weeks = None
        self.new = None
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
    # csvRowString = "Title,ArtistName,Rank,Weeks,isNew" + "\n"
    # outputFile.write(csvRowString)
    with open('../Datasets/Billboard1990AddedFeat3.csv', 'a') as outputFile: 
        writer = csv.writer(outputFile, delimiter=',', lineterminator = '\n')
        writer.writerow(["Title","ArtistName","Rank","Weeks","isNew"])
        # outputFile.write(csvRowString)
        for i in range(1000): 
            offset = random.randint(1, 520) #1508 for 1990, 468 for 2010 , 260 for 2014
            date = start_time - week*offset
            date_str = datetime.strftime(date, fmt)
            print(date_str)
            chart = billboard.ChartData('hot-100', date = date_str)
            print(chart[:5])
            for i in range(len(chart)): 
                s = chart[i]
                # csvRowString = ""
                song = Song(s.title.replace(",", " ").lower())
                if s.isNew:
                    song.new = '1' 
                else: song.new = '0' 
                song.artistName = s.artist.replace(",", " ").lower()
                song.rank = s.rank
                song.weeks = s.weeks  
                # csvRowString += song.title + "," + song.artistName + "," + str(song.rank) + ","+ str(song.weeks) + "," + song.new + "\n"
                writer.writerow([song.title,song.artistName, song.rank, song.weeks, song.new ])
                # outputFile.write(csvRowString)
            time.sleep(10)

def test():
    chart = billboard.ChartData('hot-100')
    # ChartData(name, date=None, fetch=True, timeout=25)
    # print(chart)
    for i in chart[:1]: 
        print(type(i), i.weeks, i.isNew)

    
main()
# test()