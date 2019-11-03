import spotipy
import spotipy.util as util
import csv 
import numpy as np
import pandas as pd 
import argparse 


def writing(read_file, write_file, write_file_mode, headers, cutoff, first): 
    #*************** LOADING SPOTIFY AUTHORIZATION ***************# 
    token = util.oauth2.SpotifyClientCredentials(client_id='824c8ba3f33e47898aa268ef9c7ad753', client_secret='fff42fc95229427daf9d9d26ad9da4ba')
    cache_token = token.get_access_token()
    # print(token['token_info'])
    # token._add_custom_values_to_token_info(expires_in=60)
    # _add_custom_values_to_token_info(self, token_info)
    spotify = spotipy.Spotify(cache_token)
    sp = spotipy.Spotify(auth=cache_token)

    #*************** INITIALIZING VARIABLES ********************#
    # artists = [] 
    # titles = [] 
    # isNews = [] 
    # weeks = [] 
    # ranks = [] 
    # targets = [] 
    # line_count = 0 
    # with open(read_file, "r") as f: 
    #     reader = csv.reader(f, delimiter=",", )
    #     for row in reader: 
    #         if line_count != 0: 
    #             titles.append(row[0])
    #             artists.append(row[1])
    #             ranks.append(row[2])
    #             weeks.append(row[3])
    #             isNews.append(row[4])
    #             targets.append(row[5])
    #         line_count += 1 
    
    m = cutoff 
    targets_f = [] 
    artists_f = []
    titles_f = []
    years_f = []
    ranks_f = [] 
    weeks_f = [] 
    isNews_f = [] 
    # followers = []
    # popularity = [] 
    # danceability = np.zeros([m,1])
    # energy = np.zeros([m,1])
    # key = np.zeros([m,1])
    # loudness = np.zeros([m,1])
    # mode = np.zeros([m,1])
    # speechiness = np.zeros([m,1])
    # acousticness = np.zeros([m,1])
    # instrumentalness = np.zeros([m,1])
    # liveness = np.zeros([m,1])
    # valence = np.zeros([m,1])
    # tempo = np.zeros([m,1])

    line_count = 0 
    with open(read_file, "r") as f: 
        reader = csv.reader(f, delimiter=",", )
        for row in reader: 
            if(token._is_token_expired): 
                token = util.oauth2.SpotifyClientCredentials(client_id='824c8ba3f33e47898aa268ef9c7ad753', client_secret='fff42fc95229427daf9d9d26ad9da4ba')
                cache_token = token.get_access_token()
                spotify = spotipy.Spotify(cache_token)
                sp = spotipy.Spotify(auth=cache_token)
            if line_count == 0: 
                line_count += 1
                continue
            if line_count > 10: exit 
            title,artist,rank,week,isNew,target = row[0],row[1],row[2],row[3],row[4], row[5]
            track_info = sp.search(q='artist:' + artist + ' track:' + title, type='track')
            artist_info = sp.search(q='artist:' + artist, type='artist')
            try: 
                track_id = track_info['tracks']['items'][0]['id']
                artist_id = artist_info['artists']['items'][0]
                year = track_info['tracks']['items'][0]['album']['release_date'].split('-')[0]
                month = track_info['tracks']['items'][0]['album']['release_date'].split('-')[1]
                feat = sp.audio_features(tracks=track_id)[0]
                followers = artist_id['followers']['total']
                popularity = artist_id['popularity']
                
                danceability = feat['danceability']
                energy = feat['energy']
                key= feat['key']
                loudness = feat['loudness']
                mode = feat['mode']
                speechiness = feat['speechiness']
                acousticness = feat['acousticness']
                instrumentalness = feat['instrumentalness']
                liveness = feat['liveness']
                valence = feat['valence']
                tempo = feat['tempo']
                print("inside", artist, target)
                with open(write_file, write_file_mode) as f: 
                    sp_writer = csv.writer(f, delimiter=',', lineterminator = '\n')
                    if first: sp_writer.writerow(write_headers)
                    print(artist, target)
                    sp_writer.writerow([title, artist,year,month,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo, followers, popularity, target])   

            except: print('OOOPS')
            line_count += 1

            

def main(): 
    read_file0 = '/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Combo/Only0.csv'
    read_file1 = '/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Combo/Only1.csv'
    write_headers = ['Artist','Track','Year','Month','Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo', 'Weeks', 'Rank', 'isNew', 'Target'] #'Followers'
    write_file = '/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Spotify/B+F+P.csv'
    f1 = pd.read_csv(read_file0)
    writing(read_file1, write_file, 'w+', write_headers, len(f1), True)

main()