import pandas as pd
from sqlalchemy import create_engine
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import io
from geopandas import *
from shapely.geometry import Point
import xml.etree.ElementTree as ET
from datetime import datetime


def get_quake_ml(url):
    return requests.get(url).text


def extract_events(text_response):
    root = ET.fromstring(text_response)
    ns = {'bed': 'http://quakeml.org/xmlns/bed/1.2'}

    # Extract data
    events = []
    for event in root.findall('.//bed:event', ns):
        event_id = event.get('publicID').replace('smi:ISC/evid=', '')
        event_type = event.find('bed:type', ns).text if event.find('bed:type', ns) is not None else None
        event_type_certainty = event.find('bed:typeCertainty', ns).text if event.find('bed:typeCertainty', ns) is not None else None
        origin = event.find('bed:origin', ns)
        if origin is not None:
            time = origin.find('.//bed:time/bed:value', ns).text if origin.find('.//bed:time/bed:value', ns) is not None else None
            lat = float(origin.find('.//bed:latitude/bed:value', ns).text) if origin.find('.//bed:latitude/bed:value', ns) is not None else None
            lon = float(origin.find('.//bed:longitude/bed:value', ns).text) if origin.find('.//bed:longitude/bed:value', ns) is not None else None
            depth = float(origin.find('.//bed:depth/bed:value', ns).text) if origin.find('.//bed:depth/bed:value', ns) is not None else None
        else:
            time = lat = lon = depth = None

        events.append({
            "event_id": event_id,
            "event_type": event_type,
            "event_type_certainty": event_type_certainty,
            "lon": lon,
            "lat": lat,
            "depth": depth,
            "event_dttm": time
        })

    df = pd.DataFrame(events)

    geom = [Point(xy) for xy in zip(df.lon, df.lat)]
    df = df.drop(['lon', 'lat'], axis=1)
    crs = {'init': 'epsg:4326'}
    gdf = GeoDataFrame(df, crs=crs, geometry=geom)
    df = gdf.rename({'geometry': 'loc'}, axis='columns')
    df = df.set_geometry("loc")

    return df

def extract_arrivals(text_response):
    root = ET.fromstring(text_response)
    ns = {'bed': 'http://quakeml.org/xmlns/bed/1.2'}
    
    arrivals_dict = {}
    
    for event in root.findall('.//bed:event', ns):
        event_id = event.get('publicID').replace('smi:ISC/evid=', '')
        
        for arrival in event.findall('.//bed:arrival', ns):
            phase = arrival.find('bed:phase', ns).text if arrival.find('bed:phase', ns) is not None else None
            pick_id = arrival.find('bed:pickID', ns).text if arrival.find('bed:pickID', ns) is not None else None
            
            if pick_id:
                pick = root.find(f".//bed:pick[@publicID='{pick_id}']", ns)
                pick_id = pick_id.replace('smi:ISC/pickid=', '')
                if pick is not None:
                    arrival_time = pick.find('bed:time/bed:value', ns).text if pick.find('bed:time/bed:value', ns) is not None else None
                    waveform = pick.find('bed:waveformID', ns)
                    
                    if waveform is not None:
                        network_nm = waveform.get('networkCode')
                        station_nm = waveform.get('stationCode')
                        
                        key = (event_id, station_nm, network_nm, pick_id, phase)
                        
                        if key not in arrivals_dict or (arrival_time and arrival_time < arrivals_dict[key]):
                            arrivals_dict[key] = arrival_time
    
    arrivals = [
        {
            "event_id": k[0],
            "station_nm": k[1],
            "network_nm": k[2],
            "pick_id": k[3],
            "arrival_type": k[4],
            "arrival_dttm": v
        }
        for k, v in arrivals_dict.items()
    ]
    
    df = pd.DataFrame(arrivals)
    return df


def extract_magnitudes(text_response):
    root = ET.fromstring(text_response)
    ns = {'bed': 'http://quakeml.org/xmlns/bed/1.2'}
    
    magnitudes = []
    for event in root.findall('.//bed:event', ns):
        event_id = event.get('publicID').replace('smi:ISC/evid=', '')

        for mag in event.findall('bed:magnitude', ns):
            mag_value = mag.find('bed:mag/bed:value', ns).text if mag.find('bed:mag/bed:value', ns) is not None else None
            mag_type = mag.find('bed:type', ns).text if mag.find('bed:type', ns) is not None else None
            origin_id = mag.get('publicID', '').replace('smi:ISC/magid=', '')
            

            creation_info = mag.find('bed:creationInfo', ns)
            author = creation_info.find('bed:author', ns).text if creation_info is not None and creation_info.find('bed:author', ns) is not None else None

            magnitudes.append({
                "event_id": event_id,
                "magnitude_id": origin_id,
                "mag_value": float(mag_value) if mag_value is not None else None,
                "mag_type": mag_type,
                "author_nm": author,
            })
    
    df = pd.DataFrame(magnitudes)
    return df


def extract_stations(xml_file_path):
    """Extract station information from FDSNStationXML and return as pandas DataFrame."""
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    ns = {'fdsn': 'http://www.fdsn.org/xml/station/1'}
    
    stations = []
    
    for network in root.findall('.//fdsn:Network', ns):
        network_code = network.get('code')
        
        for station in network.findall('.//fdsn:Station', ns):
            station_code = station.get('code')
            lat = station.find('fdsn:Latitude', ns).text if station.find('fdsn:Latitude', ns) is not None else None
            lon = station.find('fdsn:Longitude', ns).text if station.find('fdsn:Longitude', ns) is not None else None
            
            stations.append({
                "station_nm": station_code,
                "network_nm": network_code,
                "lat": float(lat) if lat else None,
                "lon": float(lon) if lon else None
            })
    
    df = pd.DataFrame(stations)
    return df

def get_events_and_mags(start_dt, end_dt, min_mag):
    start_dt = start_dt.split('-')
    end_dt = end_dt.split('-')
    
    # Новый URL с параметрами
    url = f'''https://www.isc.ac.uk/cgi-bin/web-db-run?request=COMPREHENSIVE&out_format=CATQuakeML&searchshape=RECT&bot_lat=&top_lat=&left_lon=&right_lon=&ctr_lat=&ctr_lon=&radius=&max_dist_units=deg&srn=&grn=&start_year={start_dt[0]}&start_month={start_dt[1]}&start_day={start_dt[2]}&start_time=00%3A00%3A00&end_year={end_dt[0]}&end_month={end_dt[1]}&end_day={end_dt[2]}&end_time=23%3A59%3A59&min_dep=&max_dep=&min_mag={min_mag}&max_mag=&req_mag_type=&req_mag_agcy=&include_links=on'''
    
    url = url.replace('\n', '').replace('        ', '')
    print(f"Запрос URL: {url}")
    
    xml = get_quake_ml(url)
    
    events = extract_events(xml)
    magnitudes = extract_magnitudes(xml)
    return events, magnitudes


def get_arrivals(start_dt, end_dt, min_mag):
    start_dt = start_dt.split('-')
    end_dt = end_dt.split('-')
    
    url = f'''https://www.isc.ac.uk/cgi-bin/web-db-run?out_format=QuakeML&request=STNARRIVALS&
        searchshape=FE&
        srn=3&
        start_year={start_dt[0]}&
        start_month={start_dt[1]}&
        start_day={start_dt[2]}&
        start_time=00:00:00&
        end_year={end_dt[0]}&
        end_month={end_dt[1]}&
        end_day={end_dt[2]}&
        end_time=23:59:59&
        min_mag={min_mag}'''
    
    url = url.replace('\n', '').replace('        ', '')
    print(url)
    return(extract_arrivals(get_quake_ml(url)))



def load_to_postgres(df, df_name):
    print(df)


    engine = create_engine('postgresql://gis:123456@10.0.62.59:55432/gis')

    try:
        df.to_sql(
            df_name,
            engine,
            schema='earthquakes_gr_low',
            if_exists='append',
            index=False,
        )
        print(f"[{df_name}] Records created successfully")
    except Exception as ex:
        print("Operation failed: {0}".format(ex))


def load_to_postgis(df, df_name):
    print(df)
    engine = create_engine('postgresql://gis:123456@10.0.62.59:55432/gis')

    try:
        df.to_postgis(
            df_name,
            engine,
            schema='earthquakes_gr_low',
            if_exists='append',
            index=False,  
        )
        print(f"[{df_name}] Records created successfully")
    except Exception as ex:
        print("Operation failed: {0}".format(ex))

def post_log(log_nm, start_dt=None, end_dt=None, load_status=None):
    
    df = pd.DataFrame({
    'log_nm': [log_nm],
    'start_dt': [start_dt],
    'end_dt': [end_dt],
    'load_status': [load_status],
    'log_dttm': [datetime.now()]
    }) 
    
    load_to_postgres(df, 'logs')


def load_events_and_mags(start_dt, end_dt, min_mag):
    # First try
    try:
        events, magnitudes = get_events_and_mags(start_dt, end_dt, min_mag)
    except Exception as e:
        post_log('events_and_mags_error', start_dt, end_dt, f'first try failed: {str(e)}')
        # Retry once
        try:
            events, magnitudes = get_events_and_mags(start_dt, end_dt, min_mag)
        except Exception as e:
            post_log('events_and_mags_error', start_dt, end_dt, f'second try failed: {str(e)}')
            return
    
    post_log('events', start_dt, end_dt, 'start')
    load_to_postgis(events, 'events')
    post_log('events', start_dt, end_dt, 'end')
    post_log('magnitudes', start_dt, end_dt, 'start')
    load_to_postgres(magnitudes, 'magnitudes')
    post_log('magnitudes', start_dt, end_dt, 'end')

def load_arrivals(start_dt, end_dt, min_mag):
    # First try
    try:
        arrivals = get_arrivals(start_dt=start_dt, end_dt=end_dt, min_mag=min_mag)
    except Exception as e:
        post_log('arrivals_error', start_dt, end_dt, f'first try failed: {str(e)}')
        # Retry once
        try:
            arrivals = get_arrivals(start_dt=start_dt, end_dt=end_dt, min_mag=min_mag)
        except Exception as e:
            post_log('arrivals_error', start_dt, end_dt, f'second try failed: {str(e)}')
            return
    
    post_log('arrivals', start_dt, end_dt, 'start')
    load_to_postgres(arrivals, 'arrivals')
    post_log('arrivals', start_dt, end_dt, 'end')



def main_load():
    date_range = pd.date_range(start='2011-03-01', end='2011-03-20', freq='M')

    for date in date_range:
        first_day = date.replace(day=1)
        last_day = date
        start_dt = f"{first_day.strftime('%Y-%m-%d')}"
        end_dt = f"{last_day.strftime('%Y-%m-%d')}"

        post_log('load_cycle', start_dt, end_dt, 'start')
        load_events_and_mags(start_dt, end_dt, 3.5)
        post_log('load_cycle', start_dt, end_dt, 'end')

def main_load_weekly():
    # Получаем все понедельники марта 2011
    mondays = pd.date_range(start='2011-03-01', end='2011-03-31', freq='W-MON')
    
    # Начало первой недели — 1 марта
    start_dates = [pd.Timestamp('2011-03-01')]
    # Конец первой недели — день перед первым понедельником (если есть)
    if len(mondays) > 0 and mondays[0] > pd.Timestamp('2011-03-01'):
        end_dates = [mondays[0] - pd.Timedelta(days=1)]
    else:
        end_dates = []
    
    # Для остальных недель start — понедельник, end — следующая неделя - 1 день
    for i in range(len(mondays)-1):
        start_dates.append(mondays[i])
        end_dates.append(mondays[i+1] - pd.Timedelta(days=1))
    # Добавляем последнюю неделю с понедельника до конца месяца
    if len(mondays) > 0:
        start_dates.append(mondays[-1])
        end_dates.append(pd.Timestamp('2011-03-31'))
    
    # Проходим по всем интервалам и вызываем загрузку
    for start_dt, end_dt in zip(start_dates, end_dates):
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')
        
        post_log('load_cycle_weekly', start_str, end_str, 'start')
        load_events_and_mags(start_str, end_str, 3.5)
        post_log('load_cycle_weekly', start_str, end_str, 'end')

main_load_weekly()


