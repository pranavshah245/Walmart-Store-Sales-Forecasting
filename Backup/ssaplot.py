# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import math

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns


def plotLRCoefs(data, constant=None, y_label=None):
    """Returns the axes of a barplot for the coefficients of a linear
    regression.


    Returns :

        g : matplotlib.axes._subplots.AxesSubplot
            Axes of the barplot


    Arguments :

        df_coefs : pandas.DataFrame
            Contains the coefficients of the LR fit
        constant : numpy.float64
            The intercept from the LR fit
        y_label : str/None
            Label for the y-axis, if non uses "Effect of Feature".
    """

    df_coefs = pd.DataFrame(data).copy()

    column_name = df_coefs.columns[0]

    # If constant is included add it to df_coefs
    if constant is not None:
        df_temp = pd.DataFrame.from_dict({'Constant': {column_name: constant}},
                                         orient='index', dtype=float)
        df_coefs = df_coefs.append(df_temp)

    # Determine the range for the color scale, center on zero
    maximum = np.amax(df_coefs.values)
    minimum = np.amin(df_coefs.values)
    maximum = max(abs(maximum), abs(minimum))
    minimum = -maximum.copy()

    # Select the color map and then normalize it based on min and max so that
    # it is centered around zero
    cmap = matplotlib.cm.autumn
    norm = matplotlib.colors.Normalize(vmin=minimum, vmax=maximum, clip=True)

    # Make a list of the colors for each feature
    colors = []
    for index, value in enumerate(df_coefs.iloc[:, 0]):
        colors.append(cmap(norm(value)))

    # Make a scatterplot and display it.
    steps = 1000
    step = (maximum-minimum)/steps
    ax = plt.scatter(np.arange(minimum, maximum+step, step),
                     np.arange(minimum, maximum+step, step),
                     c=np.arange(minimum, maximum+step, step), cmap=cmap)

    # Remove the contents of the scatterplot
    plt.clf()
    # Build the color bar based on the mapping of the scatter plot axes
    plt.colorbar(ax)

    # Build the barplot on the same axes
    ax = sns.barplot(x=df_coefs.index, y=df_coefs.iloc[:, 0], data=df_coefs,
                     palette=colors)

    plt.legend(bbox_to_anchor=(1, 1))

    # Add a y-label, either default or as an argument
    if y_label is None:
        plt.ylabel("Effect of Feature")
    else:
        plt.ylabel(y_label)

    # Do some formatting
    annotate(ax, message='Float')
    plt.xticks(rotation=-90)
    plt.tight_layout()

    # Return the axes object so that further editting can be done
    return ax


def annotate(g, location='Top', message='Count', fontsize=None):
    """Adds annotations to a seaborn.barplot based on optional arguments.

    Arguments:
        g : matplotlib.axes._subplots.AxesSubplot
            Axes object of the barplot
        location : str {'Top','Middle'}
            Determines location of the annotations
        message : str {'Count','Percent','Float'}
            Determines value to annotate.
        fontsize : int
            Overwrite built in logic for determining font size
    """

    # Calculating the tallest data point

    # Determines how far above the top of the patch the annotation is located.
    heightScaling = 1.02

    # For the Percentage message we need to know the sum of the patches
    heights = []
    for p in g.patches:
        height = p.get_height()
        heights.append(height)
    maximum = max(heights)
    total = sum(heights)

    # The user can include a font size, if they do not we scale it based
    # on the number of patches. The number has been chosen after a little
    # experimentation.
    if not fontsize:
        fontsize = str(math.floor(100./float(len(g.patches))))
#        print('The fontsize is:',fontsize)

    for p in g.patches:
        height = p.get_height()
#        print("The height is:",height, float(height),100.*height/total)

        # These are the different types of messages that we can print.
        # We need to change the formatting, and how the value is calculated
        # for each.
        if message == 'Count':
            units = ""  # Units adds a suffix to the message
            value = int(height)
            formating = '{:d}'
        elif message == 'Percentage':
            value = 100.*height/total
            units = " %"
            formating = '{:1.1f}'
        elif message == 'Float':
            units = ''
            value = float(height)
            formating = '{:1.2f}'

        # Next we determine where the annotations should be located.
        if location == 'Top':
            # If the bar is too short we need to treat it differently.
            # By experiementation I have decided that height < max/5. is too
            # short.
            max_height_cutoff = .85
            if height > maximum*max_height_cutoff:
                # if it is really tall cap its max height
                g.text(p.get_x()+p.get_width()/2.,
                       # height + 5,
                       maximum*max_height_cutoff*heightScaling,
                       formating.format(value)+units,
                       ha="center", fontsize=fontsize)
            elif height > maximum/5.:
                # if it is normal height leave it alone
                g.text(p.get_x()+p.get_width()/2.,
                       # height + 5,
                       height*heightScaling,
                       formating.format(value)+units,
                       ha="center", fontsize=fontsize)
            else:
                # then it must be short cap the min height
                g.text(p.get_x()+p.get_width()/2.,
                       maximum/5.*heightScaling,
                       formating.format(value)+units,
                       ha="center", fontsize=fontsize)
        elif location == 'Middle':
            if height > maximum/5.:

                g.text(p.get_x()+p.get_width()/2.,
                       0.5*height,
                       formating.format(value)+units,
                       ha="center", fontsize=fontsize)
            else:
                g.text(p.get_x()+p.get_width()/2.,
                       0.5*maximum/5.,
                       formating.format(value)+units,
                       ha="center", fontsize=fontsize)


if __name__ == '__main__':
    sns.set_context('talk')
    sns.set(font='Cambria', font_scale=2.0)
    sns.set_style("darkgrid")
    sns.set_context({"figure.figsize": (5, 7)})

    data = {'BooksRead': 0.14, 'IsGenderUnknown': -2,
            'IsEthnicityOther': -.5}

    data = [['BooksRead', 0.14], ['IsGenderUnknown', -2.0],
            ['IsEthnicityOther', -.5]]
    df = pd.DataFrame(data, columns=['feature', 'coef'])
    df.index = df.feature
    df.drop('feature', axis=1, inplace=True)
    print(df.head())

    plotLRCoefs(df)
    plotLRCoefs(df, constant=1.23)
