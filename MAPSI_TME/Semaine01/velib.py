import requests
import pickle as pkl
import json

ADDRESS = u'address'
ALT = u'alt'
AVAILABLE_BIKE_STANDS = u'available_bike_stands'
AVAILABLE_BIKES = u'available_bikes'
BANKING = u'banking'
BIKE_STANDS = u'bike_stands'
BONUS = u'bonus'
CONTRACT_NAME = u'contract_name'
LAST_UPDATE = u'last_update'
NAME = u'name'
NUMBER = u'number'
POSITION = u'position'
LAT = u'lat'
LNG = u'lng'
ALT = u'alt'

STATUS = u'status'
OPEN = u'OPEN'
CLOSED = u'CLOSED'

VILLE_CODE_PARIS = 75
NOMBRE_ARR_PARIS = 20
LENGTH_POSTALE = 5

def get_address(sttn_info):
    # u"57 RUE DU CHATEAU D'EAU - 75010 PARIS",
    return sttn_info[ADDRESS]
    
def get_alt(sttn_info):
    # 35.79555130004883,
    return float(sttn_info[ALT])
    
def get_available_bike_stands(sttn_info):
    # 2,
    return int(sttn_info[AVAILABLE_BIKE_STANDS])
    
def get_available_bikes(sttn_info):
    # 17,
    return int(sttn_info[AVAILABLE_BIKES])
    
def get_banking(sttn_info):
    # True,
    return bool(sttn_info[BANKING])
    
def get_bike_stands(sttn_info):
    # 19,
    return int(sttn_info[BIKE_STANDS])
    
def get_bonus(sttn_info):
    # False,
    return bool(sttn_info[BONUS])
    
def get_contract_name(sttn_info):
    # u'Paris',
    return sttn_info[CONTRACT_NAME]
    
def get_last_update(sttn_info):
    # 1410442143000,
    return long(sttn_info[LAST_UPDATE])

def get_name(sttn_info):
    # "10007 - CHATEAU D'EAU",
    return sttn_info[NAME]
def get_number(sttn_info):
    # 10007,
    return int(sttn_info[NUMBER])
    
def get_position(sttn_info):
    # {u'lat': 48.87242997325711, u'lng': 2.355489390173873}
    return sttn_info[POSITION];
    
def get_lat(sttn_info):
    return float(sttn_info[POSITION][LAT])
    
def get_lng(sttn_info):
    return float(sttn_info[POSITION][LNG])
    
def get_alt(sttn_info):
    # Altitude (Al)
    return float(sttn_info[ALT])
    
def set_alt(sttn_info, alt):
    sttn_info[ALT] = alt

def get_status(sttn_info):
    # u'OPEN'
    return sttn_info[STATUS]

def get_code_postale(sttn_info):
    address = get_address(sttn_info).strip()
    # addr_arr = address.split('-')
    # addr_arr_ville = addr_arr[-1].strip()
    # code = addr_arr_ville.split(' ')[0]
    # return int(code.strip());
    addr_arr = address.split(' ')
    for p in addr_arr:
        if len(p) == LENGTH_POSTALE and p.isdigit():
            return int(p)

def get_arrondissement(sttn_info):
    # Arrondissement(Ar)
    # int(str(get_code_postale(sttn_info))[-2:])
    return get_code_postale(sttn_info) % 100
    
def get_ville_code(sttn_info):
    # int(str(get_code_postale(sttn_info))[:2])
    return get_code_postale(sttn_info) / 1000
    
def is_paris(sttn_info):
    if get_ville_code(sttn_info) == VILLE_CODE_PARIS :
        return True
    
    return False
        
def station_is_plein(sttn_info):
    # Station pleine (Sp) [variable binaire: valeur 1 si la station est pleine]
    if get_available_bike_stands(sttn_info) > 0 :
        return False
    return True

def station_velos_disp(sttn_info):
    if get_available_bikes(sttn_info) >= 2 :
        return True
    return False

def defiler_stations(sttns_info):
    for sttn in sttns_info:
        if not is_paris(sttn):
            sttns_info.remove(sttn)
        
def get_gposition(position):
    urlGoogleAPI = "https://maps.googleapis.com/maps/api/elevation/json?locations="
    return requests.get(urlGoogleAPI+position)

def get_velibs_info(url):
    dataStation = requests.get(url)
    data = dataStation.json()

    for sttn in data:
        position = "%f,%f"%(sttn['position']['lat'],sttn['position']['lng'])
        alt = get_gposition(position)
        
        if alt.json()['status'] == "OK" :
            set_alt(sttn, alt.json()['results'][0]['elevation']) # enrichissement

def dump_velibs_info(data, fname):
    fb = open(fname,'wb')
    pkl.dump(data,fb)
    fb.close()

def load_velibs_info(fname):
    fb = open(fname,'r')
    data = pkl.load(fb) 
    fb.close
    return data;

def testcase_read_write():
    key = "6a0a07b9b956f26dba74b44e7807a4965a8ebdfd"
    data_url = "https://api.jcdecaux.com/vls/v1/stations?contract=Paris&apiKey="
    fname = "coordVelib.pkl"

    data = get_velibs_info(data_url+key)
    dump_velibs_info(data, fname)

def testcase_mise_en_forme():
    data = load_velibs_info('dataVelib.pkl')
    # Probleme : la function defiler_stations() ne peut pas enlever tous les stations dehors de Paris dans une sole fois
    print len(data)
    defiler_stations(data)
    print len(data)
    defiler_stations(data)
    print len(data)
    defiler_stations(data)
    print len(data)
    defiler_stations(data)
    print len(data)
    defiler_stations(data)
    print len(data)
    defiler_stations(data)
    print len(data)
    
def main():
    # test_read_write()
    testcase_mise_en_forme()
    
if __name__ == "__main__":
    main()
