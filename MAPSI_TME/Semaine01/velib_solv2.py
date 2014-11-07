import requests
import pickle as pkl

def get_vellibs_info(url):
    urlGoogleAPI = "https://maps.googleapis.com/maps/api/elevation/json?locations="

    dataStation = requests.get(url)
    data = dataStation.json()

    for s in data:
        position = "%f,%f"%(s['position']['lat'],s['position']['lng'])
        alt = requests.get(urlGoogleAPI+position)
        if alt.json()['status'] == "OK" :
            s[u'alt'] = alt.json()['results'][0]['elevation'] # enrichissement

def dump_vellibs_info(data, dfname):            
    fb = open(dfname,'wb')
    pkl.dump(data,fb) # penser a sauver les donnees pour eviter de refaire les operations
    fb.close()

def main():
    key = "6a0a07b9b956f26dba74b44e7807a4965a8ebdfd"
    data_url = "https://api.jcdecaux.com/vls/v1/stations?contract=Paris&apiKey="
    dfname = "coordVelib.pkl"

    data = get_vellibs_info(data_url+key)
    dump_vellibs_info(data, dfname)

if __name__ == "__main__":
    main()
