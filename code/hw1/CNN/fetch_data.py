# Fetch Bengali.AI data from Kaggle
import os

# set kaggle parameters
assert os.environ['KAGGLE_USERNAME'], "Set username with os.environ['KAGGLE_USERNAME'], obtained from Kaggle->account->Create New API Token"
assert os.environ['KAGGLE_KEY'], "Set api key with os.environ['KAGGLE_KEY'], obtained from Kaggle->account->Create New API Token"

print(os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'])
# check if running from correct path
DATA_PATH = "../Data/"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "Data/" # when executing from parent directory
assert os.path.exists(DATA_PATH), "Could not find data folder"

if not os.path.exists("{}/train.csv".format(DATA_PATH)):
    # file retrieval is not complete
    # check wheter data has already been downloaded
    if not os.path.exists("{}/bengaliai-cv19.zip".format(DATA_PATH)):
        print("Data not present, retrieving...")
        # run command to fetch data
        os.system('kaggle competitions download -c bengaliai-cv19 -p {}'.format(DATA_PATH))
        # requires pip install kaggle

    # extract the downloaded zip file, if downloaded zipped version and not extracted version
    # this depends on pip kaggle version I think
    if os.path.exists("{}/bengaliai-cv19.zip".format(DATA_PATH)):
        print("Extracting data...")
        os.system('unzip {}/bengaliai-cv19.zip -d {}'.format(
                DATA_PATH, DATA_PATH))
    if not os.path.exists("{}/train_image_data_0.parquet".format(DATA_PATH)):
        # extract zip files
        for i in range(4):
            for t in ['train', 'test']:
                os.system('unzip {}/{}_image_data_{}.parquet.zip -d {}'.format(DATA_PATH, t, i, DATA_PATH))
                os.system('rm {}/{}_image_data_{}.parquet.zip'.format(DATA_PATH, t, i))
        # also unzip train.csv
        os.system('unzip {}/train.csv.zip -d {}'.format(DATA_PATH, DATA_PATH))
        os.system('rm {}/train.csv.zip'.format(DATA_PATH))
