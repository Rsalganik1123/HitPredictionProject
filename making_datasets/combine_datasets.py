import csv
import pandas as pd 
import re
import itertools
from itertools import islice

def createSet(MSD, BB): 
    with open('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/ComboCSV.csv', 'w') as csv:
        csv.write("Title,ArtistName,Target" + "\n") 
        for track in MSD: 
            CSV_row = track+ ","+ MSD[track] + ",0" + '\n' #" 
            csv.write(CSV_row)
        for track in BB: 
            CSV_row = track+","+ BB[track] + ",1" + '\n' #",1"+ 
            csv.write(CSV_row)
def readPandas(file, mode, checkDict): 
    
    dictionary = {} 
    skipped = 0
    if (mode == 4): 
        with open(file, "r") as csv_file: 
            
            csv_reader = pd.read_csv(csv_file, error_bad_lines=True) #delimiter=','
            print('length of file: ', len(csv_reader))
            # csv_reader = csv.reader(csv_file, delimiter=',')
            print (csv_reader)
            line_count = 0 
            for line in csv_reader:
                print("LINE:", line, len(line))
                # line = line[0].split(",")
                if line_count != 0: 

                    track = line[0]
                    artist = line[1]
                    track = track.split('featuring')[0]
                    track = track.split('(')[0]
                    if track in dictionary: skipped +=1 
                    dictionary[track] = artist
                line_count += 1
        print("skipped", skipped)
    if (mode == 2): 
        with open(file, "r") as csv_file: 
            csv_reader = csv.reader(csv_file, delimiter=',')
            for line in csv_reader:
    
                track = line[0].replace("b'", "").replace("'", "").lower()
                if(track.find('//') >= 0): 
                    skipped += 1
                    continue
                    
                artist = line[1].replace("b'", "").replace("'", "").lower()
                if not track in checkDict:   
                    dictionary[track] = artist
                else:
                    if checkDict[track] != artist:
                        dictionary[track] = artist
                    else: 
                        skipped +=1 
                        # print(track, artist)
                dictionary[track] = artist
    return dictionary
def read(file, mode, checkDict, strip): 
    
    dictionary = {}
    skipped = 0
    if (mode == 4): 
        with open(file, "r") as csv_file: 
            csv_reader = csv.reader(csv_file, delimiter=',') #delimiter=','
            
            # for num, track, artist, year in csv_reader: 
            line_count = 0 
            for line in csv_reader:
                
                # print("LINE:", line, len(line))
                # line = line[0].split(",")
                if line_count != 0: 

                    track = line[0]
                    artist = line[1]
                    track = track.split('featuring')[0]
                    track = track.split('(')[0]
                    if track in dictionary: skipped +=1 
                    dictionary[track] = artist
                line_count += 1
        print("skipped", skipped)
    if (mode == 2): 
        skipped = 0 
        line_count = 0 
        with open(file, "r") as csv_file: 
            csv_reader = csv.reader(csv_file, delimiter=',')
            for line in csv_reader:
                # if line_count == 0: 
                #     line_count+=1 
                #     continue
                
                track = line[0].replace("b'", "").replace("'", "").lower()
                if(track.find('//') >= 0): 
                    skipped += 1
                    continue
                if 'Title' in track: continue 
                artist = line[1].replace("b'", "").replace("'", "").lower()
                if track in checkDict:   
                    dictionary[track] = artist
                    skipped += 1
                    continue 
                if track in dictionary: 
                    if dictionary[track] == artist:
                        skipped += 1 
                        # print(track,artist) 
                        continue 
                else: 
                    dictionary[track] = artist
        print("skipped", skipped)
    return dictionary



def main(): 
    
    dictMSD = read('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/MSDSubsetCSV.csv', 4, {}, [])
    # dictMSD = readPandas('../Datasets/MSDSubsetCSV.csv', 4, {})
    print("wrote MSD songs", len(dictMSD))
    # print(dictMSD['deep sea creature'])
    dictBillboard = read('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/BillboardCSV1990.csv', 2, dictMSD, [])
    # dictBillboard = readPandas('../Datasets/BillboardCSV1990.csv', 2, dictMSD)
    print("wrote bill songs", len(dictBillboard))
    # dictMSD.update(dictBillboard)
    createSet(dictMSD, dictBillboard)
    # checkFormatting(dictMSD)

    csv = pd.read_csv("../Datasets/ComboCSV.csv")
    billboard = csv.loc[csv['Target'] == 1]
    not_billboard = csv.loc[csv['Target'] == 0]
    
    print("billboard", len(billboard))
    print("not_billboard", len(not_billboard))
    print('total', len(csv))
    
main() 
