import urllib

def get_vellibs_info(url):
    fb = urllib.urlopen(url)
    data = fb.read()
    fb.close()
    return data

def dump_vellibs_info(data, dfname):
    fb = open(dfname,'wb')
    fb.write(data)
    fb.close()

def main():
    key = "6a0a07b9b956f26dba74b44e7807a4965a8ebdfd"
    data_url = "https://api.jcdecaux.com/vls/v1/stations?contract=Paris&apiKey="
    dfname = "./dump.solution1.data"
    data = get_vellibs_info(data_url+key)
    dump_vellibs_info(data, dfname)
    
if __name__ == "__main__":
    main()
