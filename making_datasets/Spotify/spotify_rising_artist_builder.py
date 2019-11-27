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
    
    m = cutoff 
    targets_f = [] 
    artists_f = []

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
            if line_count > 2000: break 
            artist,target = row[0].strip(),row[1]
            
            try: 
                artist_tracks = sp.search(q='artist:' + artist, type='track')
                artist_info = sp.search(q='artist:' + artist, type="artist")
                title = artist_tracks['tracks']['items'][0]['name']
                track_info = sp.search(q='artist:' + artist + ' track:' + title, type='track')
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
                # print("inside", artist, target)
                with open(write_file, write_file_mode) as f: 
                    sp_writer = csv.writer(f, delimiter=',', lineterminator = '\n')
                    if first: 
                        sp_writer.writerow(headers)
                        first = False 
                    print(line_count, artist, target)
                    sp_writer.writerow([title, artist,year,month,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo, followers, popularity, target])   
                    line_count += 1
            except: print('something went wrong loading', artist)
            
def test(): 
    token = util.oauth2.SpotifyClientCredentials(client_id='824c8ba3f33e47898aa268ef9c7ad753', client_secret='fff42fc95229427daf9d9d26ad9da4ba')
    cache_token = token.get_access_token()
    spotify = spotipy.Spotify(cache_token)
    sp = spotipy.Spotify(auth=cache_token)
    artist = 'dua lipa'
    artist_info = sp.search(q='artist:' + artist, type='track') 
    print(artist_info['tracks']['items'][0]['name'])       

def main(): 
    read_file0 = '/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Billboard/Rising_artists_target.csv'
    write_headers = ['Artist','Track','Year','Month','Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo',  'Followers', 'Popularity', 'Target'] #'Followers'
    write_file = '/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Spotify/Rising.csv'
    f1 = pd.read_csv(read_file0)
    # writing(read_file1, write_file, 'a+', write_headers, len(f1), True)
    writing(read_file0, write_file, 'a+', write_headers, len(f1), True)

#main()
test()