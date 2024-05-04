import numpy as np
import pandas as pd
from glob import glob
import os
from tqdm import tqdm
tqdm.pandas()

'''Read data and check'''
df = pd.read_csv('../data/train.csv')
print(df.shape)   # (115488, 3)
print(df.columns)   # ['id', 'class', 'segmentation']
print(df.id.unique().shape)   # (38496,)

'''Split the id column into case, day, slice'''
df['case'] = df['id'].apply(lambda x: int(x.split('_')[0].replace('case', '')))
df['day'] = df['id'].apply(lambda x: int(x.split('_')[1].replace('day', '')))
df['slice'] = df['id'].apply(lambda x: x.split('_')[3])

path = '../data/train'
images_paths = glob(os.path.join(path, '**', '*.png'), recursive=True)
first_4 = images_paths[0].rsplit('/', 4)[0]

sub_path = []
for i in range(df.shape[0]):
    sub_path.append(os.path.join(
        first_4,
        "case"+str(df["case"].values[i]),
        "case"+str(df["case"].values[i]) + "_" + "day"+str(df["day"].values[i]),
        "scans",
        "slice_"+str(df["slice"].values[i]))
    )

df['sub_path'] = sub_path

sub_path = []
for i in range(0, len(images_paths)):
    sub_path.append(str(images_paths[i].rsplit("_", 4)[0]))

sub_df = pd.DataFrame({
    "sub_path": sub_path,
    'path': images_paths
})

data = df.merge(sub_df, on='sub_path')
data.drop('sub_path', axis=1, inplace=True)

data['width'] = data["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
data['height'] = data["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))

print(data.head(5))

'''Re-organize train data'''
df = data.copy()
final_df = pd.DataFrame({'id': df['id'][::3]})

final_df['large_bowel'] = df['segmentation'][::3].values
final_df['small_bowel'] = df['segmentation'][1::3].values
final_df['stomach'] = df['segmentation'][2::3].values

col = ['path', 'case', 'day', 'slice', 'width', 'height']

for col in col:
    final_df[col] = df[col][::3].values

final_df.reset_index(inplace=True, drop=True)
final_df.fillna('', inplace=True)
final_df['count'] = np.sum(final_df.iloc[:, 1:4] != '', axis=1).values

print(final_df.shape)
print(final_df.columns)

'''
shape (38496, 11)
['id', 'large_bowel', 'small_bowel', 'stomach', 'path',
'case', 'day', 'slice', 'width', 'height', 'count']
'''

'''Save the final_df'''
final_df.to_csv('../code/final_df.csv')