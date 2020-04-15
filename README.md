# DeepLearningProject

## Retrieve data
The data can be downloaded directly from Kaggle to any/most systems. This required the package `kaggle` to be installed, which can be done with `pip install kaggle`. On Google Colab this is installed by default. 
To retrieve the data, the username and API key need to be provided. These can be obtained from Kaggle.com -> account -> Create New API Token, which downloads a json file with the username and key. Set these before running the script with `os.environ['KAGGLE_USERNAME'] = 'username'` and `os.environ['KAGGLE_KEY'] = 'ajasdhflkjashdlfjkahsl'`. 
Finally, retrieve the data with the command `python CNN/fetch_data.py`, which can be executed from a Jupyter Notebook using `!python CNN/fetch_data.py` or from Python code using `os.system('python CNN/fetch_data.py')`. 
