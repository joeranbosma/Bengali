import os

if not os.path.exists("CNN"):
    os.environ['KAGGLE_USERNAME'] = "username" # username from the json file
    os.environ['KAGGLE_KEY'] = "keyasdfasdf" # key from the json file

    os.system('git init .')
    os.system('git remote add -t \* -f origin https://username:dummypassword@github.com/L-Hess/CS4180-Deep-Learning--CNN.git')
    os.system('git checkout master')


if not os.path.exists("Data/train.csv"):
    os.system('python CNN/fetch_data.py')

# install Weights & Biases
os.system('pip install wandb -q --upgrade')
os.system('wandb login keyasdkjfhaslkjdfh')
            
# setup figure params

# further settings
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 4.5)
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["savefig.dpi"] = 400
plt.rcParams["savefig.transparent"] = True
plt.rcParams.update({'font.size': 14})
plt.rcParams["savefig.bbox"] = 'tight'
