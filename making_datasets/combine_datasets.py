import csv
import pandas as pd 
import re
import itertools
from itertools import islice

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

def createSet(data, mode, isMSD): 
    with open('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Only' + str(mode)+'N.csv', 'a+') as f:
        writer = csv.writer(f, delimiter=',', lineterminator = '\n')
        # csv.write("Title,ArtistName,Rank,Weeks,isNew,Target" + "\n") 
        writer.writerow(["Title","ArtistName","Rank","Weeks","isNew","Target"])
         
        for song in data: 
            #Title, ArtistName, Rank, Weeks,isNew,Target
            # song = Song(val[0])
            # song.artistName = val[1]
            # song.rank = 0
            # song.new = '1'
            # song.weeks = 0
            target = str(mode)
            writer.writerow([song.title,song.artistName, song.rank, song.weeks, song.new,target])
            # CSV_row = val[0]+ ","+ val[1] + ","+ '0' + ","+ '0' + ","+ str(mode) + '\n' 
            # csv.write(CSV_row)
         
            # for song in data: 
            #     #Title, ArtistName, Rank, Weeks,isNew,Target
            #     # print(val)
            #     # song = s.title
            #     # song.artistName = val[1]
            #     # song.rank = val[2]
            #     # song.new = val[3]
            #     # song.weeks = val[4]
            #     target = str(mode)
            #     writer.writerow([song.title,song.artistName, song.rank, song.weeks, song.new,target])
            #     # CSV_row = val[0]+ ","+ val[1] + ","+ str(val[2]) + ","+ str(val[3]) + ","+ str(mode) + '\n' 
            #     # csv.write(CSV_row)

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
            artist = artist.split('(')[0]
            song = Song(track)
            song.artist = artist
            if mode == 1: 
                song.rank = line[2]
                song.new = line[3]
                song.weeks = line[4]
            else: 
                song.rank = 0 
                song.new = 1
                song.weeks = 0
            if song not in this_list and song not in checkList: 
                this_list.append(song) 
            else: skipped +=1 
    return this_list


def main(): 
    
    dictMSD = read2('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/MSD/MSDSubsetCSV.csv', 4, [])
    print("wrote MSD songs", len(dictMSD))

    dictBillboard = read2('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Billboard/Billboard1990AddedFeat3.csv', 2, dictMSD)
    print("wrote bill songs", len(dictBillboard))
    
    createSet(list(set(dictBillboard)), 1, False)
    createSet(list(set(dictMSD)), 0, True)
    

    # createSet2(dictMSD, dictBillboard)
    
    # csv0 = pd.read_csv("/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Only0N.csv")
    # csv1 = pd.read_csv("/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Only1N.csv")
    # billboard = csv1.loc[csv1['Target'] == 1]
    # not_billboard = csv0.loc[csv0['Target'] == 0]
    
    # print("billboard", len(billboard))
    # print("not_billboard", len(not_billboard))
    # print('total', len(csv0) + len(csv1))
    
main() 
