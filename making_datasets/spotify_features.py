import spotipy
import spotipy.util as util
import csv 
import numpy as np
import pandas as pd 


def main(): 
    skipped = 0 
    token = util.oauth2.SpotifyClientCredentials(client_id='824c8ba3f33e47898aa268ef9c7ad753', client_secret='fff42fc95229427daf9d9d26ad9da4ba')
    cache_token = token.get_access_token()
    spotify = spotipy.Spotify(cache_token)
    sp = spotipy.Spotify(auth=cache_token)
    
    # extra_token = spotipy.Spotify(token[''])
    billboard_written = 0 
    non_billboard_written = 0 
    artist_l = [] 
    track_l = [] 
    target_l = [] 
    with open('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/ComboCSV.csv', "r") as f: 
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                # print(row[0])
                # print(row[1])
                track_l.append(row[0]) 
                artist_l.append(row[1])
                target_l.append(row[2])
            line_count += 1 

    m = len(artist_l)
    target_fl = [] 
    artist_fl = []
    track_fl = []
    year_fl = []
    month_fl  = []
    #score_fl = []
    #label_fl = []
    followers = []
    danceability = np.zeros([m,1])
    energy = np.zeros([m,1])
    key = np.zeros([m,1])
    loudness = np.zeros([m,1])
    mode = np.zeros([m,1])
    speechiness = np.zeros([m,1])
    acousticness = np.zeros([m,1])
    instrumentalness = np.zeros([m,1])
    liveness = np.zeros([m,1])
    valence = np.zeros([m,1])
    tempo = np.zeros([m,1])


    j = 0
    counter = 0 
    for i in range(0,m):
        if(token._is_token_expired): 
            # token = util.oauth2.SpotifyClientCredentials(client_id='824c8ba3f33e47898aa268ef9c7ad753', client_secret='fff42fc95229427daf9d9d26ad9da4ba')
            # cache_token = token.get_access_token()
            # spotify = spotipy.Spotify(cache_token)
            # sp = spotipy.Spotify(auth=cache_token)
            refresh()

        artist = artist_l[i]
        track = track_l[i]
        #score = score_l[i]
        #label = label_l[i]
        # if mode == 'create_feature': 
        #     artist_info = sp.search(q='artist:' + artist, type='artist')
        #     follower_count = artist_info['artists']['items'][0]['followers']['total']
        #     followers.append(follower_count)
        #     print('fc', follower_count)
        track_info = sp.search(q='artist:' + artist + ' track:' + track, type='track')
        artist_info = sp.search(q='artist:' + artist, type='artist')
        track_id = track_info['tracks']
        artist_id = artist_info['artists']
        track_id2 = track_id['items']
        artist_id2 = artist_id['items']
        if track_id2 != []:
            if artist_id2 != []: 
                follower_count = artist_id2[0]['followers']['total']
                followers.append(follower_count)
                print('fc', artist_l[i], follower_count)
            else: followers.append(0)
            counter += 1
            year = track_info['tracks']
            year_1 = year['items']
            year_2 = year_1[0]
            year_3 = year_2['album']
            year_4 = year_3['release_date']
            year_5 = year_4.split('-')
            if len(year_5) > 1:
                year_6 = year_5[0]
                track_id3 = track_id2[0]
                track_id4 = track_id3['id']
                month = year_5[1]
                try: 
                    feat_t = sp.audio_features(tracks=track_id4)
                
                    if(len(feat_t) > 0): 
                        feat = feat_t[0]
                        print(i, artist_l[i], target_l[i])
                        artist_fl.append(artist_l[i])
                        track_fl.append(track_l[i]) 
                        target_fl.append(target_l[i])
                        if target_l[i] == '1': billboard_written += 1
                        if target_l[i] == '0': non_billboard_written += 1
                        year_fl.append(year_6)
                        month_fl.append(month)
                        #score_fl.append(score_l[i])
                        #label_fl.append(label_l[i])
                        danceability[j] = feat['danceability']
                        energy[j] = feat['energy']
                        key[j] = feat['key']
                        loudness[j] = feat['loudness']
                        mode[j] = feat['mode']
                        speechiness[j] = feat['speechiness']
                        acousticness[j] = feat['acousticness']
                        instrumentalness[j] = feat['instrumentalness']
                        liveness[j] = feat['liveness']
                        valence[j] = feat['valence']
                        tempo[j] = feat['tempo']
                        j += 1
                except: 
                    print(track, artist, "was skipped")
                    skipped += 1
    with open('/Users/Owner/Desktop/School/2019-2020/COMP400/Code/Datasets/Combo+Spotify+Followers.csv', "w") as f:
        sp_writer = csv.writer(f, delimiter=',', lineterminator = '\n')
        sp_writer.writerow(['Artist','Track','Year','Month','Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo', 'Followers','Target'])
        for i in range(0,j-1):
            # print(artist_fl[i], track_fl[i], target_l[i][0])
            sp_writer.writerow([artist_fl[i],track_fl[i],year_fl[i],month_fl[i],danceability[i][0],energy[i][0],key[i][0],loudness[i][0],mode[i][0],speechiness[i][0],acousticness[i][0],instrumentalness[i][0],liveness[i][0],valence[i][0],tempo[i][0], followers[i], target_l[i][0]]) 
    # else:         
    #     with open('../Datasets/Combo+Spotify.csv', "w") as f: 
    #         sp_writer = csv.writer(f, delimiter=',', lineterminator = '\n')
    #         sp_writer.writerow(['Artist','Track','Year','Month','Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo', 'Target'])
    #         for i in range(0,j-1):
    #             # print(artist_fl[i], track_fl[i], target_l[i][0])
    #             sp_writer.writerow([artist_fl[i],track_fl[i],year_fl[i],month_fl[i],danceability[i][0],energy[i][0],key[i][0],loudness[i][0],mode[i][0],speechiness[i][0],acousticness[i][0],instrumentalness[i][0],liveness[i][0],valence[i][0],tempo[i][0], target_l[i][0]])
            
    print("billboard has ", billboard_written, "entries")
    print("non billboard has ", non_billboard_written, "entries")
    print('skipped', skipped)
def test(): 
    # token = util.oauth2.SpotifyClientCredentials(client_id='824c8ba3f33e47898aa268ef9c7ad753', client_secret='fff42fc95229427daf9d9d26ad9da4ba')
    sp_auth = util.oauth2.SpotifyOAuth(client_id='824c8ba3f33e47898aa268ef9c7ad753', client_secret='fff42fc95229427daf9d9d26ad9da4ba', redirect_uri='127.0.0.1/callback')
    token_info = sp_auth.get_cached_token()
    if not token_info: 
        # auth_url = sp_auth.get_authorize_url()
        # print(auth_url)
        # response = input('Paste the above link into your browser, then paste the redirect url here: ')
        code = sp_auth.parse_response_code("https://example.com/v1/refresh")
        token_info = sp_auth.get_access_token(code)

        token = token_info['access_token']
        
    if sp_auth._is_token_expired: 
        token_info = sp_auth.refresh_access_token(token_info['refresh_token'])
        print("WAS EXPIRED, NOW", sp_auth._is_token_expired)
    
    token_info = sp_auth.get_cached_token()
    
    

    sp = spotipy.Spotify(auth=token) 
    # if( not token._is_token_expired): print("NOT EXPIRED")

    found = sp.search(q='artist:' + "Selena Gomez", type='artist')
    items = found['artists']['items'][0]['followers']['total']#['external_urls']
           
def test2(): 
    csv = pd.read_csv("../Datasets/Combo+Spotify+Followers.csv")
    billboard = csv.loc[csv['Target'] == 1]
    not_billboard = csv.loc[csv['Target'] == 0]
    
    print("billboard", len(billboard))
    print("not_billboard", len(not_billboard))
    print('total', len(csv))
# main()
test() 