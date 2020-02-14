### Call this from another file with:
### from Plot_to_tex import Plot_to_tex as plt_tex
### multiple_y_series = np.zeros((nrOfDataSeries,nrOfDataPoints), dtype=int); # actually fill with data
### lineLabels = [] # add a label for each dataseries
### plt_tex.plotMultipleLines(plt_tex,single_x_series,multiple_y_series,"x-axis label [units]","y-axis label [units]",lineLabels,"4b",4)
### 4b=filename
### 4 = position of legend, e.g. top right.
###
### For a single line, use:
### plt_tex.plotSingleLine(plt_tex,range(0, len(dataseries)),dataseries,"x-axis label [units]","y-axis label [units]",lineLabel,"4b",4)
import random
from matplotlib import pyplot as plt
from matplotlib import lines
import matplotlib.pyplot as plt

class Plot_to_tex:

    # plot graph (legendPosition = integer 1 to 4)
    def plotSingleLine(self,x_path,y_series,x_axis_label,y_axis_label,label,filename,legendPosition):
        fig=plt.figure();
        ax=fig.add_subplot(111);
        ax.plot(x_path,y_series,c='b',ls='-',label=label,fillstyle='none');
        plt.legend(loc=legendPosition);
        plt.xlabel(x_axis_label);
        plt.ylabel(y_axis_label);
        plt.savefig('../../latex/hw1/images/'+filename+'.png');
        plt.show();

    # plot graphs
    def plotMultipleLines(self,x,y_series,x_label,y_label,label,filename,legendPosition):
        fig=plt.figure();
        ax=fig.add_subplot(111);

        # generate colours
        cmap = self.get_cmap(len(y_series[:,0]))

        # generate line types
        lineTypes = self.generateLineTypes(y_series)

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