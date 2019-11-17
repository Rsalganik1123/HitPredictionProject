import csv
import pandas as pd 
import re
import itertools
from itertools import islice

artists = {} 
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

def createSet(data, mode, writeMode): 
    with open('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Combo/Only' + str(mode)+'.csv', writeMode) as fn:
        writer_csv = csv.writer(fn, delimiter=',', lineterminator = '\n')
        # fn.write("Title,ArtistName,Rank,Weeks,isNew,Target" + "\n") 
       
        writer_csv.writerow(["Title","ArtistName","Ranks","Weeks","isNew","Target"])
         
        for node in data: 
            target = str(mode)
            writer_csv.writerow([node[0],node[1], node[2], node[3], node[4],target])

def read2(file, mode, checkList): 
    this_list = [] 
    skipped = 0 
    with open(file, "r") as f: 
        csv_reader = csv.reader(f, delimiter=',')
        for line in csv_reader: 
            track = line[0]
            artist = line[1]
            if "Title" in track or 'ArtistName' in artist: 
                skipped +=1 
                continue 
            track = track.split('Featuring')[0]
            track = track.split('(')[0]
            
            artist = artist.split('Featuring')[0]
            artist = artist.split('featuring')[0]
            artist = artist.split('+')[0]
            artist = artist.split('(')[0]
            artist = artist.split('&')[0]
            song = Song(track)
            song.artistName = artist
            
            if mode == 1: 
                song.rank = line[2]
                song.weeks = line[3]
                song.new = line[4]
                artists[artist] = 1 
            else: 
                song.rank = 0
                if artist in artists: 
                    print(artist)
                    song.new = 0 
                else: song.new = 1
                song.weeks = 0
            node = (song.title, song.artistName, song.rank, song.weeks, song.new)
            if node not in this_list and node not in checkList: 
                this_list.append(node) 
            else: skipped +=1 
    print(skipped)
    return this_list


def main(): 

    dictBillboard = read2('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Billboard/Billboard1990AddedFeat3.csv', 1, [])
    print("wrote bill songs", len(list(set(dictBillboard))))
    
    dictMSD = read2('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/MSD/MSDSubsetCSV.csv', 0, dictBillboard)
    print("wrote MSD songs", len(list(set(dictMSD))))
    
    createSet(dictBillboard, 1, "w+")
   
    createSet(dictMSD, 0, "a+")


    

    # createSet2(dictMSD, dictBillboard)
    
    # csv0 = pd.read_csv("/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Only0N.csv")
    # csv1 = pd.read_csv("/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Only1N.csv")
    # billboard = csv1.loc[csv1['Target'] == 1]
    # not_billboard = csv0.loc[csv0['Target'] == 0]
    
    # print("billboard", len(billboard))
    # print("not_billboard", len(not_billboard))
    # print('total', len(csv0) + len(csv1))
    
main() 
