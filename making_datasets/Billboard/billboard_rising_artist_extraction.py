import billboard
import csv
import time 

class Artist:
    # songCount = 0
    # songDictionary = {}

    def __init__(self, name):
        
        self.name = name
        self.rank = None
        self.weeks = None 
        self.isNew = None 
        

def load():
    first = False
    with open('./Datasets/Billboard/Rising_artists.csv', "a+") as f: 
        writer = csv.writer(f, delimiter=',', lineterminator = '\n')
        if first: writer.writerow(["Date", "ArtistName","Rank", "Weeks", "isNew"])
        chart = billboard.ChartData('emerging-artists', date = '2017-08-19')
        while '2016' not in chart.previousDate: 
            chart = billboard.ChartData('emerging-artists', date = chart.previousDate)
            
            print(chart.previousDate, chart[0])
            for v in chart:
                artist = Artist(v) 
                artist.isNew = v.isNew
                artist.rank = v.rank
                artist.weeks = v.weeks 
        
                writer.writerow([chart.previousDate, artist.name,artist.rank, artist.weeks, artist.isNew])
            time.sleep(10)    
def test():
    chart = billboard.ChartData('emerging-artists', date = '2017-08-12')
    print(chart)
#load()
test()