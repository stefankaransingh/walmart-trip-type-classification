import numpy as np
import pandas as pd
from local import *
import requests
from tqdm import tqdm

API_KEYS = [API_KEY_2,API_KEY_1,API_KEY_3]
URL = 'http://api.walmartlabs.com/v1/items?'
RATE_LIMIT = 3000

def checksum(x):
    """
    Calculate the checksum
    """
    try:
        odd = map(int, str(x)[-1::-2])
        even = map(int, str(x)[-2::-2])
        sum_odd3 = sum(odd) * 3
        total = sum_odd3 + sum(even)
        rem = total % 10
        if rem == 0:
            return rem
        return 10 - rem
    except:
        return -9999

def get_full_upc(x,only_pad=False):
    try:
        if len(str(x)) == 12:
            return x
        else:
            if only_pad == False:
                x = str(x) + str(checksum(x))
            if len(str(x)) < 12:
                missing = 12 - len(str(x))
                zeros = '0' * missing
                xx = zeros + str(x)
                return xx
            elif len(str(x)) == 12:
                return x
            else:
                return None

    except Exception as ex:
        return -9999

if __name__ =='__main__':
    try:
        write_to_data = pd.read_csv('data/clean_data_write_to.csv')
        for api_key in API_KEYS:
            payload = {
                'apiKey': api_key,
                 'upc':None
                }
            print("Total No. of extracted data: ",len(write_to_data[write_to_data['dataScraped'] == True]))
            print("___start____")
            requests_made = 0
            pbar = tqdm(total=RATE_LIMIT)
            for index,row in write_to_data[np.isnan(write_to_data['dataScraped']) | write_to_data['dataScraped']==False].iterrows():
                not_scraped = np.isnan([row['dataScraped']])[0]
                try:
                    if not_scraped or row['dataScraped'] == False:
                        decoded_upc = str(get_full_upc(row['DecodedUpc'],True))
                        payload['upc'] = decoded_upc
                        response = requests.get(URL,params=payload)
                        response_data = response.json()
                        try:
                            write_to_data.set_value(index,'salePrice',response_data["items"][0]["salePrice"])
                        except KeyError:
                            pass
                        try:
                            write_to_data.set_value(index,'name',response_data["items"][0]["name"])
                        except KeyError:
                            pass

                        try:
                            write_to_data.set_value(index,'brandName',response_data["items"][0]["brandName"])
                        except KeyError:
                            pass

                        try:
                            write_to_data.set_value(index,'categoryPath',response_data["items"][0]["categoryPath"])
                            write_to_data.set_value(index,'dataScraped',True)
                            requests_made += 1
                            pbar.update(requests_made)
                        except KeyError:
                            pass

                        try:
                            write_to_data.set_value(index,'offerType',response_data["items"][0]["offerType"])
                        except KeyError:
                            pass
                except KeyboardInterrupt:
                    write_to_data.to_csv('data/clean_data_write_to.csv',index=False)
                    print("Total No. of extracted data: ",len(write_to_data[write_to_data['dataScraped'] == True]))
                except Exception as ex:
                    write_to_data.set_value(index,'dataScraped',None)
                    pass
            print("Total No. of extracted data: ",len(write_to_data[write_to_data['dataScraped'] == True]))
            write_to_data.to_csv('data/clean_data_write_to.csv',index=False)
    except Exception as ex:
        write_to_data.to_csv('data/clean_data_write_to.csv',index=False)
