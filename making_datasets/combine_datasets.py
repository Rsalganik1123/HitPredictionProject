import csv
import pandas as pd 
import re
import itertools
from itertools import islice


def createSet2(MSD, BB): 
    with open('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/ComboCSV.csv', 'w') as csv:
        csv.write("Title,ArtistName,Target" + "\n") 
        for val in MSD: 
            CSV_row = val[0]+ ","+ val[1] + ",0" + '\n' 
            csv.write(CSV_row)
        for val in BB: 
            CSV_row = val[0]+ ","+ val[1] + ",1" + '\n'
            csv.write(CSV_row)
def createSet(data, mode): 
    with open('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Only' + str(mode)+'.csv', 'a+') as csv:
        csv.write("Title,ArtistName,Target" + "\n") 
        for val in data: 
            CSV_row = val[0]+ ","+ val[1] + ","+ str(mode) + '\n' 
            csv.write(CSV_row)

def read2(file, mode, checkList): 
    tup_list = [] 
    skipped = 0 
    with open(file, "r") as f: 
        csv_reader = csv.reader(f, delimiter=',')
        for line in csv_reader: 
            track = line[0]
            artist = line[1]
            if "Title" in track or 'ArtistName' in artist: 
                skipped +=1 
                continue 
            track = track.split('featuring')[0]
            track = track.split('(')[0]
            artist = artist.split('featuring')[0]
            artist = artist.split('(')[0]
            if (track,artist) not in tup_list and (track, artist) not in checkList: 
                tup_list.append((track,artist)) 
            else: skipped +=1 
    return tup_list


def main(): 
    
    dictMSD = read2('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/MSDSubsetCSV.csv', 4, [])
    print("wrote MSD songs", len(dictMSD))

    dictBillboard = read2('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/BillboardCSV1990.csv', 2, dictMSD)
    print("wrote bill songs", len(dictBillboard))
    
    createSet(dictMSD, 0)
    createSet(dictBillboard, 1)

    # createSet2(dictMSD, dictBillboard)
    
    csv0 = pd.read_csv("/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Only0.csv")
    csv1 = pd.read_csv("/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Only1.csv")
    billboard = csv1.loc[csv1['Target'] == 1]
    not_billboard = csv0.loc[csv0['Target'] == 0]
    
    print("billboard", len(billboard))
    print("not_billboard", len(not_billboard))
    print('total', len(csv0) + len(csv1))
    
main() 
