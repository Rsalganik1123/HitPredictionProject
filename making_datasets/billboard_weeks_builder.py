import csv
import billboard 

def load(): 
    first = False
    with open('./Datasets/Billboard/ContinuousChart.csv', 'r') as f: 
        #writer = csv.writer(outputFile, delimiter=',', lineterminator = '\n')
        #if first: writer.writerow(["Title","ArtistName","Rank","Weeks","isNew","Target"])                
        
def main(): 
    #Load the values into dict by week 
    #If the next week's dict contains same entry, give this one a plus one
    #  