import os 
import pandas as pd 


def save_cnf_zipped_data_to_csv(data, filename):
    if len(data) == 0:
        raise ValueError('No data to save')

    df = []
    for item in data:    
        instance_id = item[0]
        variables = item[1][0]
        clauses = item[1][1]
        df.append([instance_id, variables, clauses])
    
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    pd.DataFrame(df, columns=['instance_id', 'variables', 'clauses']).to_csv(filename, index=False)
