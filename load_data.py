import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

'''Load the data'''
api = KaggleApi()
api.authenticate()
dataset_name = 'uw-madison-gi-tract-image-segmentation'
data_path = './data'

# Create 'data' directory if it doesn't exist
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Download dataset
api.competition_download_files(dataset_name, path=data_path)

zip_path = os.path.join(data_path, 'uw-madison-gi-tract-image-segmentation.zip')
extract_path = data_path

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

