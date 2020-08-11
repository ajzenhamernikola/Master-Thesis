import pandas as pd


def get_column_without_duplicates(data: pd.DataFrame, col_name: str):
    col = []
    n = data.count()[0]

    for i in range(n):
        row = data.iloc[i]
        inst = row[col_name]

        if inst not in col:
            col.append(inst)

    return col 


def sort_by_array_order(column: pd.Series, arr: list):
    result = []
    for val in column.values:
        result.append(arr.index(val))
    return pd.Series(result)
