import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
import sys
import re

# Load data
train_data = pd.read_parquet('dataset.parquet')

# Preprocessing
train_data['created'] = pd.to_datetime(train_data['created'])
train_data['updated'] = pd.to_datetime(train_data['updated'])

# add column called updated_ts from updated by calling timestamp() method on each line
train_data['updated_ts'] = train_data['updated'].apply(lambda x: x.timestamp())
train_data['created_ts'] = train_data['created'].apply(lambda x: x.timestamp())

# remove lines with created_ts less than 10000
train_data = train_data[train_data['created_ts'] > 10000]
train_data = train_data[train_data['updated_ts'] > 10000]

# normalize created_ts column to have values between 0 and 1
train_data['created_ts_n'] = (train_data['created_ts'] - train_data['created_ts'].min()) / (train_data['created_ts'].max() - train_data['created_ts'].min())
train_data['updated_ts_n'] = (train_data['updated_ts'] - train_data['updated_ts'].min()) / (train_data['updated_ts'].max() - train_data['updated_ts'].min())

# a function that takes string like "18°54'2"E 47°27'32"N" to get the first part
# and convert coordinate to float like 18.900555555555556
def get_lonlat(lonlat):
    def get_lonlat_parts(lonlat):
        lonlat = lonlat.replace('-', '')
        return re.findall(r'(\d+)°(\d+)\'(\d+)\"([EWNS])', lonlat)[0]
    try:
        # print(lonlat)
        # if '-' in lonlat:
        #     breakpoint()
        # breakpoint()
        lon, lat = lonlat.split()
        lon = list(get_lonlat_parts(lon))
        lat = list(get_lonlat_parts(lat))

        # cast to float lon[0:3] and lat[0:3]
        lon[0:3] = [float(x) for x in lon[0:3]]
        lat[0:3] = [float(x) for x in lat[0:3]]

        lon_float = (lon[0] + lon[1] / 60 + lon[2] / 3600) * (-1 if lon[3] in 'WS' else 1)
        lat_float = (lat[0] + lat[1] / 60 + lat[2] / 3600) * (-1 if lat[3] in 'WS' else 1)
        return lon_float, lat_float
    except:
        print("NONE")
        return None, None

train_data['lon'] = train_data['lonlat'].apply(lambda x: get_lonlat(x)[0])
train_data['lat'] = train_data['lonlat'].apply(lambda x: get_lonlat(x)[1])

train_data['lon'] = train_data['lon'] / 180
train_data['lat'] = train_data['lat'] / 90


train_data.to_parquet('dataset_processed.parquet')
