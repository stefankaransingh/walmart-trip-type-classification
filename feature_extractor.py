import numpy as np
import pandas as pd

DATA_FILE ='data/clean_data_v3.csv'
WRITE_TO_FILE = 'data/company_feature_extracted.csv'

if __name__ == '__main__':
    data = pd.read_csv(DATA_FILE)
    companies = set(data['Company'])
    visits = {}
    for index,row in data.iterrows():
        if row['VisitNumber'] not in visits.keys():
            visits[row['VisitNumber']] = {
                                          'Company':dict.fromkeys(companies,0),
                                          }

        visits[row['VisitNumber']]['Company'][row['Company']] += row['ScanCount']

    print("____part_1_done_____")

    d = {'VisitNumber':[]}

    print("__creating_dictionary__")

    for company in companies:
        d.update({company:[]})

    for visit_id in visits.keys():
        d['VisitNumber'].append(visit_id)

        for company in companies:
            d[company].append(visits[visit_id]['Company'][company])

    feature_extracted_data = pd.DataFrame(data=d)
    print("Total No. of features: ", len(feature_extracted_data.columns))

    feature_extracted_data.to_csv(WRITE_TO_FILE,index=False)
    print("___done__")
