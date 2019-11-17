import sys
import os
import glob
import hdf5_getters
import re

class Song:
    songCount = 0
    # songDictionary = {}

    def __init__(self, songID):
        self.id = songID
        Song.songCount += 1
        self.artistName = None
        self.title = None
        self.year = None

def main():
    # print("we in")
    outputFile1 = open('../Datasets/MSDSubsetCSV.csv', 'w')
    csvRowString = ""
 
    csvRowString = "Title,ArtistName"
    csvAttributeList = re.split(',', csvRowString)
    for i, v in enumerate(csvAttributeList):
        csvAttributeList[i] = csvAttributeList[i].lower()
    csvRowString += ",\n"

    basedir = '/Users/Owner/Desktop/School/2019-2020/COMP400/MillionSongSubset/'
    ext = ".h5" 

    #FOR LOOP
    for root, dirs, files in os.walk(basedir):        
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            print (f)
            songH5File = hdf5_getters.open_h5_file_read(f)
            song = Song(str(hdf5_getters.get_song_id(songH5File)))
            
            song.title = str(hdf5_getters.get_title(songH5File)).replace("b'", "").lower()
            song.artistName = str(hdf5_getters.get_artist_name(songH5File)).replace("b'", "").lower()
            song.year = str(hdf5_getters.get_year(songH5File))
            if(int(song.year) < 1990): 
                print('nope', int(song.year))
                continue

            for attribute in csvAttributeList:
                # print "Here is the attribute: " + attribute + " \n"

                if attribute == 'ArtistName'.lower():
                    csvRowString += "\"" + song.artistName.replace("'", "") +"\""  #took out   "\"" before and after                
                elif attribute == 'Title'.lower():
                    csvRowString += "\"" +song.title.replace("'", "")  + "\"" 
                else:
                    csvRowString += "Erm. This didn't work. Error. :( :(\n"

                csvRowString += ","

            #Remove the final comma from each row in the csv
            lastIndex = len(csvRowString)
            csvRowString = csvRowString[0:lastIndex-1]
            csvRowString += "\n"
            outputFile1.write(csvRowString)
            csvRowString = ""

            songH5File.close()

    outputFile1.close()
	
main()