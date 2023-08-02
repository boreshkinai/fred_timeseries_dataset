from fred import Fred

import logging
import os
from collections import OrderedDict
from typing import Dict, List
from typing import NamedTuple

import fire
import numpy as np
import pandas as pd
import patoolib
import pathlib
import pickle
import json
import time
from tqdm.autonotebook import tqdm

from common.metrics import smape_2
from common.settings import STORAGE_DIR
from common.utils import download_url, url_file_name


FRED_STORAGE = os.path.join(STORAGE_DIR, 'fred')
FRED_CATEGORIES_CACHE = os.path.join(FRED_STORAGE, 'categories.pickle')
FRED_META_CACHE = os.path.join(FRED_STORAGE, 'ts_meta.pickle')
KEY_FILE = os.path.join(FRED_STORAGE, 'keys.txt')
OBSERVATIONS_FILE_RAW = os.path.join(FRED_STORAGE, 'raw_observations.h5')
FRED_INFO_FILE_PATH = os.path.join(FRED_STORAGE, 'FREDInfo.csv')
FRED_TEST_CACHE_FILE_PATH = os.path.join(FRED_STORAGE, 'fred-train.npz')
FRED_TRAIN_CACHE_FILE_PATH = os.path.join(FRED_STORAGE, 'fred-test.npz')
H5_COMPRESSION_LEVEL = 9
H5_COMPRESSION_LIB = "bzip2"
RESPONSE_LIMIT = 1000

FRED_SP = {'Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily'}

# This is same as for M4
FRED_HORIZONS_MAP = {
    'Yearly': 6,
    'Quarterly': 8,
    'Monthly': 18,
    'Weekly': 13,
    'Daily': 14,
}

FRED_FREQ_SHORT_TO_M4_FREQ_MAP = {
    'A': 'Yearly',
    'Q': 'Quarterly',
    'M': 'Monthly',
    'W': 'Weekly',
    'D': 'Daily',
}

# This is same as for M4
FRED_SEASONALITY_MAP = {
    'Yearly': 1,
    'Quarterly': 4,
    'Monthly': 12,
    'Weekly': 1,
    'Daily': 1,
}

# This is same as for M4
FRED_MIN_LENGTH = {
    'Yearly': 19,
    'Quarterly': 24,
    'Monthly': 60,
    'Weekly': 93,
    'Daily': 107,
}


def get_api_key():
    if not os.path.isfile(KEY_FILE):
        print(f"key file should be stored in {KEY_FILE} path")
        print("The key file is a json file and the key is stored in the format \{\"fred\": \"key_sequence\"\}.")
        print("A key can be ontained by registering here: https://research.stlouisfed.org/useraccount/apikey")
        raise Exception("FRED key file is not found")
        
    with open(KEY_FILE) as json_file:
        return json.load(json_file)['fred']
    
    
def get_fred_api():
    api_key = get_api_key()
    return Fred(api_key=api_key,response_type='df')


def get_category_children(parent, max_try=20, wait_delay=20.0):
    """
    Fetch children for a given category, if fetching too many, wait
    """
    
    logging.info(f'Process parent category {parent}')
    
    while max_try > 0:
        try:
            return get_fred_api().category.children(parent)
        except Exception as exc:
            logging.info(f"Caught Error processing '{parent}': {exc}")
            if "too many requests" in str(exc).lower():
                ## Too many requests
                logging.info(f'Too Many Requests, Waiting {wait_delay} seconds to retry, retries: {max_try}') 
                time.sleep(wait_delay)
            else:
                ## Unknown error, try to sleep
                logging.info(f'Unknown error, Waiting {wait_delay} seconds to retry, retries: {max_try}') 
                time.sleep(wait_delay)
        max_try -= 1
    raise Exception("Maximum retries exceeded")


def get_all_fred_children_categories(parent: int = 0): 
    children = get_category_children(parent)
    if len(children) == 0:
        logging.info(f'Processed category {parent}')
        return [parent]
    else:
        ids = list(children.id)
        for id in children.id:
            ids += get_all_fred_children_categories(parent = id)
    return ids


def get_fred_categories_cached():
    if os.path.exists(FRED_CATEGORIES_CACHE):
        with open(FRED_CATEGORIES_CACHE, 'rb') as f:
            logging.info(f'Loading FRED categories from cache: {FRED_CATEGORIES_CACHE}')
            fred_categories = pickle.load(f)
    else:
        logging.info(f'Loading categories...')
        fred_categories = get_all_fred_children_categories(0)
        logging.info(f'Loaded {len(fred_categories)} categories')
        fred_categories = list(set(fred_categories))
        logging.info(f'Number of unique categories: {len(fred_categories)}')
        with open(FRED_CATEGORIES_CACHE, 'wb') as f:
            pickle.dump(fred_categories, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fred_categories


def get_category_series(category, params, max_try=20, wait_delay=20.0):
    """
    Fetch time series metadata for a given category, if fetching too many, wait
    """
    fr = get_fred_api()
    while max_try > 0:
        try:
            return fr.category.series(category, params=params)
        except Exception as exc:
            logging.info(f"Caught Error processing '{category}': {exc}")
            if "too many requests" in str(exc).lower():
                ## Too many requests
                logging.info(f'Too Many Requests, Waiting {wait_delay} seconds to retry, retries: {max_try}') 
                time.sleep(wait_delay)
            else:
                ## Unknown error, try to sleep
                logging.info(f'Unknown error, Waiting {wait_delay} seconds to retry, retries: {max_try}') 
                time.sleep(wait_delay)
        max_try -= 1
    raise Exception("Maximum retries exceeded")


def load_ts_meta_per_category(category):
    cache_fname = FRED_META_CACHE.split('.')
    cache_fname.insert(-1, f'_category{category}')
    cache_fname.insert(-1, '.')
    cache_fname = "".join(cache_fname)
    
    if os.path.exists(cache_fname):
        with open(cache_fname, 'rb') as f:
            ts_meta = pickle.load(f)
    else:
        fr = get_fred_api()
        offset = 0
        ts_meta = []
        df = get_category_series(category, params={'limit':RESPONSE_LIMIT, 'offset': offset})
        ts_meta.append(df)
        while len(df) == RESPONSE_LIMIT:
            offset += RESPONSE_LIMIT
            df = get_category_series(category, params={'limit':RESPONSE_LIMIT, 'offset': offset})
            ts_meta.append(df)
        ts_meta = pd.concat(ts_meta)
        
        with open(cache_fname, 'wb') as f:
            pickle.dump(ts_meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    return ts_meta


def load_ts_meta(fred_categories: List[int] = None):
    ts_meta = {}
    for idx, category in enumerate(fred_categories):
        ts_meta[category] = load_ts_meta_per_category(category)
        logging.info(f'Loaded {len(ts_meta[category])} time-series meta-data in category {category}, {idx} out of {len(fred_categories)} done...')
    ts_meta = pd.concat(ts_meta)
    # This is to account for the fact that same time series can be part of several categories
    ts_meta = ts_meta.groupby('id').head(1).reset_index().set_index('id')
    ts_meta.drop(['level_0', 'level_1'], inplace=True, axis=1)
    return ts_meta


def get_ts_first_release_by_id(ts_id: str = None):
    fr = get_fred_api()
    ts = fr.series.observations(ts_id, params={"output_type": 1, "realtime_start": "1776-07-04"})
    first_release = ts.groupby('date').head(1)
    ts = first_release.set_index('date')['value']
    return ts


def get_ts_latest_release_by_id(ts_id: str = None):
    fr = get_fred_api()
    ts = fr.series.observations(ts_id)
    ts = ts.set_index('date')['value']
    return ts


def get_observations(seriesid, max_try=50, wait_delay=20.0):
    """Fetch the first-released observations of a FRED series given its id
    
    The function will automatically retry a certain number of times 
    if the FRED fetch function returns an error, waiting `wait_delay` 
    between requests (default=10.0 seconds) with a maximum `max_try` number of 
    times (default=50).
    The return value is a Pandas Series (float64), or an exception is
    raised if the request cannot be successfully completed. 
    The NaT entries in the values column are substituted for NaN.
    """
    fred = get_fred_api()
    getter = get_ts_first_release_by_id
    while max_try > 0:
        try:
            data = getter(seriesid)
            data[data.isna()] = np.nan  # Catch the NaT which may occur
            data = data.astype("float64")
            return data
        except Exception as exc:
            logging.info(f"Caught Error processing '{seriesid}': {exc}") # logging.info
            if ("The series does not exist in ALFRED but may exist in FRED" in str(exc)) \
                or ("this exceeds the maximum number of vintage dates allowed" in str(exc).lower()) \
                or ("bad request" in str(exc).lower()):
                ## There are a couple of situations where ALFRED (vintage data)
                ## would not work properly
                getter = get_ts_latest_release_by_id
            elif "out of bounds nanosecond timestamp" in str(exc).lower():
                ## Some series like HPGDPUKA (GDP in the UK) start before 1600
                ## which does not seem to be supported by Pandas. Return an empty
                ## DataFrame for these (`None` is not supported by HDF5).
                return pd.DataFrame()
            elif "too many requests" in str(exc).lower():
                ## Too many requests or some such
                logging.info(f'Too Many Requests, Waiting {wait_delay} seconds to retry, retries: {max_try}') 
                time.sleep(wait_delay)
            else:
                ## Unknown error, try to sleep
                logging.info(f'Unknown error, Waiting {wait_delay} seconds to retry, retries: {max_try}') 
                time.sleep(wait_delay)
        max_try -= 1
    raise Exception("Maximum retries exceeded")


def load_ts_observations(ts_ids):
    ## And produce the output in Append mode. This is idempotent, and if the 
    ## download fails for any reason, we can simply restart it from where it 
    ## left off by re-invoking the command
    with pd.HDFStore(OBSERVATIONS_FILE_RAW, mode="a", complevel=H5_COMPRESSION_LEVEL, 
                     complib=H5_COMPRESSION_LIB) as h5store:
        
        for id in tqdm(ts_ids, miniters=1, total=len(ts_ids)):
            if id not in h5store:
                data = get_observations(id)
                h5store.append(id, data)


def check_observations(obs, min_length, gt):
    """Should we keep this Pandas.Series based on length and value criteria.
    """
    obs_nona = obs.dropna()
    keep = (obs_nona.shape[0] >= min_length)
    if gt is not None:
        keep = (keep and (obs_nona > gt).all())
    return keep


def save_dataset_cache(ts_meta):    
    meta_clean_m4_style = []
    valid_ts = {}
    count_na = 0
    count_valid = 0
    with pd.HDFStore(OBSERVATIONS_FILE_RAW, mode="r") as h5store:
        for id, ts_meta_row in tqdm(ts_meta.iterrows(), miniters=1, total=len(ts_meta)):
            if ts_meta_row.frequency_short in FRED_FREQ_SHORT_TO_M4_FREQ_MAP.keys():

                if id not in h5store:
                    logging.info(f'Time series ID {id} is not in the raw download file {OBSERVATIONS_FILE_RAW}')
                    continue

                sp = FRED_FREQ_SHORT_TO_M4_FREQ_MAP[ts_meta_row.frequency_short]
                ts = h5store[id]

                # Check if time-series is valid
                if not check_observations(ts, min_length=FRED_MIN_LENGTH[sp], gt=0.0):
                    continue
                if ts.isna().any():
                    count_na += 1
                    continue
                count_valid += 1
                # Save valid time-series
                valid_ts[id] = ts
                # Process meta-data into M4-style table
                row_meta = pd.DataFrame(columns=["category", "Frequency", "Horizon", "SP", "StartingDate"], 
                                        data=[["unknown", FRED_SEASONALITY_MAP[sp], FRED_HORIZONS_MAP[sp], sp, ts.index.min()]], index=[id])
                row_meta.index.name = "FREDid"
                meta_clean_m4_style.append(row_meta)

    logging.info(f'Number of time series with NaNs: {count_na}')
    logging.info(f'Number of valid time-series: {count_valid}')
    
    # Create tran/test split
    meta_clean_m4_style = pd.concat(meta_clean_m4_style)
    train_split = OrderedDict(list(zip(meta_clean_m4_style.index, [[]] * len(meta_clean_m4_style))))
    test_split = OrderedDict(list(zip(meta_clean_m4_style.index, [[]] * len(meta_clean_m4_style))))
    for id, id_meta in tqdm(meta_clean_m4_style.iterrows(), miniters=1, total=len(meta_clean_m4_style)):
        h = id_meta["Horizon"]
        ts = valid_ts[id].values
        values_train = ts[:-h]
        values_test = ts[-h:]
        train_split[id] = values_train
        test_split[id] = values_test
    # Save data
    meta_clean_m4_style.to_csv(FRED_INFO_FILE_PATH)
    np.array(list(train_split.values())).dump(FRED_TRAIN_CACHE_FILE_PATH)
    np.array(list(test_split.values())).dump(FRED_TEST_CACHE_FILE_PATH)
    
    
def load_fred_info():
    return pd.read_csv(FRED_INFO_FILE_PATH)
            
    
class FREDDataset(NamedTuple):
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True):
        fred_info = load_fred_info()
        return FREDDataset(ids=fred_info.FREDid.values,
                           groups=fred_info.SP.values,
                           frequencies=fred_info.Frequency.values,
                           horizons=fred_info.Horizon.values,
                           values=np.load(
                               FRED_TRAIN_CACHE_FILE_PATH if training else FRED_TEST_CACHE_FILE_PATH,
                               allow_pickle=True))
    
    def to_training_subset(self):
        return self.to_hp_search_training_subset(horizon=0)
    
    def to_hp_search_training_subset(self, horizon: float = 1):
        values = []
        for i, v in enumerate(self.values):
            if horizon == 0:
                final_point = None
            else:
                final_point = -int(horizon*self.horizons[i])
            values.append(v[:final_point])
        return FREDDataset(ids=self.ids,
                           groups=self.groups,
                           frequencies=self.frequencies,
                           horizons=self.horizons,
                           values=np.array(values))
    
    def to_hp_search_validation_subset(self, horizon: float = 1):
        values = []
        for i, v in enumerate(self.values):
            if horizon == 1:
                final_point = None
            else:
                final_point = -int((horizon-1)*self.horizons[i])
            values.append(v[-int(horizon*self.horizons[i]):final_point])
                
        return FREDDataset(ids=self.ids,
                           groups=self.groups,
                           frequencies=self.frequencies,
                           horizons=self.horizons,
                           values=np.array(values))
    
    
class FredSummary:
    def __init__(self, dataset: FREDDataset = None):
        self.target_values = {}
        self.groups = np.unique(dataset.groups)
        self.group_idxs = {}
        for group in self.groups:
            group_idxs = np.argwhere(np.isin(dataset.groups, [group])).ravel()
            self.target_values[group] = np.array([dataset.values[i] for i in group_idxs])
            self.group_idxs[group] = group_idxs
            
    def evaluate(self, forecast: np.array) -> Dict[str, float]:
        results = OrderedDict()
        cumulative_metrics = 0
        cumulative_points = 0
        offset = 0
        for group in self.groups:
            group_target = self.target_values[group]
            group_forecast = np.array([forecast[i] for i in self.group_idxs[group]])

            group_smape = smape_2(group_forecast, group_target)
            cumulative_metrics += np.sum(group_smape)
            
            cumulative_points += np.prod(group_target.shape)
            results[group] = round(float(np.mean(group_smape)), 2)
            
        results['Average'] = round(cumulative_metrics / cumulative_points, 2)
        return results
    
    
def init():
    pathlib.Path(FRED_STORAGE).mkdir(parents=True, exist_ok=True)
    
    fred_categories = get_fred_categories_cached()
    logging.info(f'Following {len(fred_categories)} FRED categories loaded: {fred_categories}')
    ts_meta = load_ts_meta(fred_categories)
    logging.info(f'The number of time-series meta-data records loaded: {len(ts_meta)}')
    
    unique_ts_ids = list(ts_meta.index)
    logging.info(f'The number of of unique records is: {len(unique_ts_ids)}')
    
    logging.info(f'Downloading time series observations...')
    load_ts_observations(ts_ids=unique_ts_ids)
    
    logging.info(f'Cleaning and saving dataset...')
    save_dataset_cache(ts_meta)
    

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    fire.Fire()