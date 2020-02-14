import random
from matplotlib import pyplot as plt
from matplotlib import lines
import matplotlib.pyplot as plt


import numpy as np

#execute create a line
def do_run_a():
    results = [1,2,3,4,5]
    return results;

# Generate multiple lines and plot them
def do4b():
    # perform computation
    for run in range(10):
        res = do_run_a() # create array of lines

    # generate plot data
    plotResult = np.zeros((10,len(res)), dtype=int);
    lineLabels = []

    # perform computation
    for run in range(10):
        res = do_run_a() # create array of lines

        # store computation data for plotting
        lineLabels.append(f'Run {run}')
        plotResult[run,:]=res;

    # Command to plot multiple lines
    plotMultipleLines(range(0, len(res)),plotResult,"[runs]]","fitness [%]",lineLabels,"4b",4)
    

# plot graph (legendPosition = integer 1 to 4)
def plotSingleLine(x_path,y_series,x_axis_label,y_axis_label,label,filename,legendPosition):
    fig=plt.figure();
    ax=fig.add_subplot(111);
    ax.plot(x_path,y_series,c='b',ls='-',label=label,fillstyle='none');
    plt.legend(loc=legendPosition);
    plt.xlabel(x_axis_label);
    plt.ylabel(y_axis_label);
    plt.savefig('../../latex/hw1/images/'+filename+'.png');
    plt.show();

# plot graphs
def plotMultipleLines(x,y_series,x_label,y_label,label,filename,legendPosition):
    fig=plt.figure();
    ax=fig.add_subplot(111);

    # generate colours
    cmap = get_cmap(len(y_series[:,0]))

    # generate line types
    lineTypes = generateLineTypes(y_series)

    for i in range(0,len(y_series)):
        # overwrite linetypes to single type
        lineTypes[i] = "-"
        ax.plot(x,y_series[i,:],ls=lineTypes[i],label=label[i],fillstyle='none',c=cmap(i)); # color

    # configure plot layout
    plt.legend(loc=legendPosition);
    plt.xlabel(x_label);
    plt.ylabel(y_label);
    plt.savefig('../../latex/hw1/images/'+filename+'.png');

# Generate random line colours
# Source: https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def generateLineTypes(y_series):
    # generate varying linetypes
    typeOfLines = list(lines.lineStyles.keys())

    while(len(y_series)>len(typeOfLines)):
        print("Adding line")
        typeOfLines.append("-.");

    # remove void lines
    for i in range(0, len(y_series)):
        if (typeOfLines[i]=='None'):
            typeOfLines[i]='-'
        if (typeOfLines[i]==''):
            typeOfLines[i]=':'
        if (typeOfLines[i]==' '):
            typeOfLines[i]='--'
    return typeOfLines

if __name__ == '__main__':
    print("now running 4a")
    res = do_run_a()
    plotSingleLine(range(0, len(res)),res,"[runs]]","fitness [%]","run 1","4a",4)

    print("now running 4b")
    do4b()