import pandas as pd 


def get_column_without_duplicates(data, col_name):
    col = []
    n = data.count()[0]

    for i in range(n):
        row = data.iloc[i]
        inst = row[col_name]

        if inst not in col:
            col.append(inst)

    return col 

