import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline


file_names = ['result_non_classif.csv','result.csv']

def read_and_concat_data(file_name):
    for i in range(len(file_name)):
        if i == 0:
            data = pd.read_csv(file_name[i],header=0)
            data['type'] = 'without classifier'
        else:
            data1 = pd.read_csv(file_name[i],header=0)
            data1['type'] = 'with classifier'
            data = pd.concat([data,data1])
    return data


if __name__ == '__main__':
    
    print ("Start drawing...")
    plot_data = read_and_concat_data(file_names)
    col_data = plot_data.columns[1::2] 
    col_std = plot_data.columns[2::2]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    num_points = 300  # Specify the number of data points after interpolation
    for i in range(2):
        for j in range(2):
            for t in plot_data['type'].unique():
                type_data = plot_data[plot_data['type'] == t]
                
                # Interpolate using UnivariateSpline
                x = type_data['num_samples']  
                y = type_data[col_data[i*2+j]]
                y_err = type_data[col_std[i*2+j]]
                
                
                spl = UnivariateSpline(x, y, s=0.00001) # s is the smoothing parameter
                spl_err = UnivariateSpline(x, y_err, s=0.00001) # s is the smoothing parameter
                # f_err = interp1d(x, y_err, kind='cubic', bounds_error=False, fill_value="extrapolate")
                x_new = np.linspace(x.min(), x.max(), num_points)
                y_new = spl(x_new)
                y_err = spl_err(x_new)
                
                
                # Calculate the error band
                y_lower = y_new - y_err
                y_upper = y_new + y_err
                
                # Draw the curve
                sns.lineplot(x=x_new, y=y_new, ax=axes[i, j], label=t)
                axes[i, j].fill_between(x_new, y_new - y_err, y_new + y_err, alpha=0.3)
            
            axes[i, j].set_title(col_data[i*2+j])
            # plt.legend(title='type')
    plt.show()