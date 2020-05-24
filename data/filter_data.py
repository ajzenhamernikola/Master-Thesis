import os 
import shutil
import pandas as pd

cnf_files = {}
this_dirname = os.path.dirname(__file__)
cnf_dirname = os.path.join(this_dirname, 'cnf')

for _, _, files in os.walk(cnf_dirname):
    for file in files:
        cnf_files[file] = pd.DataFrame({'benchmark': [file]}, index=[0])

def extract_data_from_csv(input_file, output_file):
    csv_file = os.path.join(this_dirname, 'metadata', input_file)
    data = pd.read_csv(csv_file)

    name_without_ext = input_file
    ext_idx = name_without_ext.rfind('.csv')
    if ext_idx != -1:
        name_without_ext = name_without_ext[:ext_idx]

    def change_name(row):
        row['benchmark'] = row['benchmark'].strip()
        
        slash_idx = row['benchmark'].rfind('/')
        if slash_idx != -1:
            new_name = row['benchmark'][slash_idx+1:]
            row['benchmark'] = new_name

        cnf_idx = row['benchmark'].rfind('.cnf')
        if cnf_idx == -1:
            raise ValueError('Benchmark does not have .cnf in its name!')
        new_name = row['benchmark'][:cnf_idx+4]
        row['benchmark'] = new_name

        return row

    data = data.transform(change_name, axis=1)

    def keep_if_file_exists(df):
        benchmarks = df['benchmark']
        labels = []
        for b in benchmarks:
            if b in cnf_files:
                df1 = cnf_files[b]
                if name_without_ext not in df1:
                    df2 = pd.DataFrame({name_without_ext: [True]}, index=[0])
                    cnf_files[b] = pd.concat([df1, df2], axis=1, sort=False)
                labels.append(True)
            else:
                labels.append(False)
        return labels

    data = data.loc[keep_if_file_exists, :]

    csv_file = os.path.join(this_dirname, 'metadata', output_file)
    data.to_csv(csv_file, index=False)

tracks = ['main', 'random']
for track in tracks:
    extract_data_from_csv(track + '.csv', track + '_extracted.csv')

df = None
for cnf in cnf_files:
    if df is None:
        df = cnf_files[cnf]
    else:
        df = df.append(cnf_files[cnf], sort=False, ignore_index=True)

df = df.fillna(False)
csv_file = os.path.join(this_dirname, 'metadata', 'benchmark_track_extracted.csv')
df.to_csv(csv_file, index=False)

criterion = None
for track in tracks:
    if criterion is None:
        criterion = (df[track] == True)
    else:
        criterion = (criterion & (df[track] == True))
have_cnfs_in_both_tracks = df[criterion].empty
print('There are' + (' not' if have_cnfs_in_both_tracks else '') + ' instances in both track')

for track in tracks:
    track_folder = os.path.join(this_dirname, 'cnf', track)
    if not os.path.exists(track_folder):
        os.makedirs(track_folder)
    
    instances = df[['benchmark', track]]
    for idx in instances.index:
        benchmark = df.iloc[idx]['benchmark']
        is_in_track = df.iloc[idx][track]
        if is_in_track:
            src = os.path.join(this_dirname, 'cnf', benchmark)
            shutil.copy2(src, track_folder)