# Decipher Bengali Graphemes

This project employs Convolutional Neural Networks to decipher Bengali graphemes, as part of the Bengali.AI challenge on Kaggle, https://www.kaggle.com/c/bengaliai-cv19.  

## Report
A report of our most important findings is available at [Report Bengali Competition.pdf](https://github.com/joeranbosma/Bengali/blob/master/Report%20Bengali%20Competition.pdf). 

## Retrieve data
The data can be downloaded directly from Kaggle to any/most systems. This required the package `kaggle` to be installed, which can be done with `pip install kaggle`. On Google Colab this is installed by default.  

To retrieve the data, the username and API key need to be provided. These can be obtained from Kaggle.com -> account -> Create New API Token, which downloads a json file with the username and key. Set these before running the script with `os.environ['KAGGLE_USERNAME'] = 'username from file'` and `os.environ['KAGGLE_KEY'] = 'kaggle key from file'`.  

Finally, retrieve the data with the command `python CNN/fetch_data.py`, which can be executed from a Jupyter Notebook using `!python CNN/fetch_data.py` or from Python code using `os.system('python CNN/fetch_data.py')`. 

This gives:  

```
os.environ['KAGGLE_USERNAME'] = 'username from file'
os.environ['KAGGLE_KEY'] = 'kaggle key from file'
os.system('python CNN/fetch_data.py')
```
