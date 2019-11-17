import csv
import pandas as pd 



def write(first, rising_artists, billboard_artists):
    count = 0 
    count_ones = 0 
    with open('./Datasets/Billboard/Rising_artists_target.csv', "a+") as f: 
        writer_csv = csv.writer(f, delimiter=',', lineterminator = '\n')
        if first: writer_csv.writerow(["ArtistName", "Target"])
        for a in rising_artists: 
            target = 0 
            if a in billboard_artists: 
                target = 1
                count_ones += 1
            writer_csv.writerow([a, target])
            count +=1 
    print('written', count, count_ones )
        
def load_billboard_artists(): 
    billboard_artists = [] 
    with open('./Datasets/Billboard/ContinuousChart.csv', "r") as f: 
        csv_reader = csv.reader(f, delimiter=",")
        for line in csv_reader: 
            artist = line[2]
            billboard_artists.append(artist)
    return billboard_artists

def load_rising_artists():  
    artists = [] 
    with open('./Datasets/Billboard/Rising_artists_continuous.csv', "r") as f: 
        csv_reader = csv.reader(f, delimiter=",")
        for line in csv_reader: 
            artist = line[1].lower() 
            if 'Artist' in artist: continue 
            if artist not in artists: 
                artists.append(artist)
            else: continue 
    return artists 
    
def main(): 
    csv = pd.read_csv('./Datasets/Billboard/Rising_artists_continuous.csv')
    rising_artists = load_rising_artists()
    print("loaded", len(rising_artists), 'artists') 
    billboard_artists = load_billboard_artists() 
    print(billboard_artists[:3])
    write(True, rising_artists, billboard_artists) 

main() 