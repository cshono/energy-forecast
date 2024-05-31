import gridstatus
import pandas as pd
import numpy as np 

# Define Params 
FILE_DF_TRAIN_EXPORT = "./data/df_train.csv" 
START_DATE = "2022-01-01"
END_DATE = "2023-07-15" 
LOCATIONS = ["SHILOH3_7_N002", "SNJSEA_1_N101"] 
TARGET_LOCATION = "SHILOH3_7_N002"
MARKET = "DAY_AHEAD_HOURLY" 
TARGET_COL = "LMP" 
FILES_WEATHER = {
    "sd": "./data-raw/sd_2022.csv"
    , "la": "./data-raw/la_2022.csv"
    , "sf": "./data-raw/sf_2022.csv" 
}
WEATHER_COLS = {
    "DATE": "datetime"
    , "HourlyDryBulbTemperature": "temperature"
    , "HourlyRelativeHumidity": "relativeHumidity"
    , "HourlyWindSpeed": "windSpeed" 
}
MAX_INTERP_HRS = 4
LAGS = [48] # ,48]

def convert_to_float(raw): 
    clean = raw.replace('s','') 
    clean = clean.replace('V','') 
    if raw == '': 
        return np.nan 
    return float(clean) 

WEATHER_COL_CONVERTERS = {
    "HourlyDryBulbTemperature": convert_to_float
    , "HourlyRelativeHumidity": convert_to_float
    , "HourlyWindSpeed": convert_to_float
}

# Query LMP 
try: 
    df_lmp = pd.read_pickle("./data-raw/df_lmp_raw.pkl") 
    #print(this_will_bread) 
except: 
    caiso = gridstatus.CAISO() 
    start = pd.Timestamp(START_DATE).normalize()
    end = pd.Timestamp(END_DATE).normalize()
    df_lmp = caiso.get_lmp(
        start=start, end=end, market=MARKET, locations=LOCATIONS, sleep=5
    )
    df_lmp.to_pickle("./data-raw/df_lmp_raw.pkl") 


# Clean LMP data 
df_lmp['datetime'] = pd.to_datetime(df_lmp['Time']) 
df_lmp = df_lmp.set_index("datetime").drop("Time", axis=1) 
df_lmp.index = df_lmp.index.tz_convert("Etc/GMT+8") # Also PST does not exist in Python boo.... (despite EST and MST existing) 
df_lmp = df_lmp.loc[df_lmp.Location == TARGET_LOCATION, TARGET_COL] 

# Read Historical Weather 
df_train = df_lmp.copy() 
for station, filename in FILES_WEATHER.items(): 
    df_weather = pd.read_csv(
        filename
        , usecols=WEATHER_COLS.keys() 
        , converters=WEATHER_COL_CONVERTERS
    ).rename(
        columns = WEATHER_COLS
    )

    # Clean Weather data 
    df_weather['datetime'] = pd.to_datetime(df_weather.datetime)
    df_weather = df_weather.set_index('datetime') #.drop("DATE", axis=1) 
    df_weather.index = df_weather.index.tz_localize(tz='Etc/GMT+8') # We like this because there is not ambiguity or gaps in the time series 
    df_weather = df_weather.resample("H").mean().interpolate(limit=MAX_INTERP_HRS)
    df_weather.columns = [f'{c}_{station}' for c in df_weather.columns] 
    """
    Issues with the raw data: 
    - need to figure out timezone, (including distinction between local and local standard time)
    - LET THEM HAVE THE ERROR MESSAGE FOR NON-EXISTENT OR AMBIGUOUS TIMESTAMPS 
    - use datetime data type for convenient functionality 
    - numeric columns have values that can't convert to float '*s' or '' 
    - data not on a normalized time-interval 
    """

    # Merge Datasets 
    df_train = pd.concat([df_train, df_weather], axis=1) 

# Extract Features
df_train["hour"] = df_train.index.hour
df_train["month"] = df_train.index.month 

for lag in LAGS: 
    df_train[f'{TARGET_COL}_lag{lag}'] = df_train[TARGET_COL].shift(lag)

# Export Data 
df_train.to_csv(FILE_DF_TRAIN_EXPORT) 