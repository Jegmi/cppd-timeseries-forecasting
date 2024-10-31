import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay
#from sklearn.metrics import auc # ?

def my_auc(recall,precision):
    return np.sum(precision[:-1] * np.abs(np.diff(recall)))

def plot_auc_for_val(tmin, tmax, ax1, ax2, predictions, y_val, hours_y_val):    
    mask = (hours_y_val[:,0] < tmax) & (hours_y_val[:,0] > tmin)

    # norm
    pred = (predictions[mask, 0] - 1) / 3
    true = (y_val[mask,0] - 1) / 3 
    
    # Compute precision-recall pairs and area under precision-recall curve
    precision, recall, _ = precision_recall_curve(true > 0, pred)
    pr_auc = my_auc(recall, precision)
    
    # Compute false positive rate and true positive rate for ROC, and AUC-ROC score
    fpr, tpr, _ = roc_curve(true > 0, pred)
    roc_auc = roc_auc_score(true > 0, pred)
        
    # Plot Precision-Recall Curve
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=ax1)
    ax1.set_ylim([0, 1.05])
    ax1.set_title(f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    
    # Plot ROC Curve
    roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    roc_disp.plot(ax=ax2)
    ax2.set_title(f'ROC curve (AUC = {roc_auc:.2f})')


def arr_to_pcolor_plt(arr, colors = [(0, 'gray'), (.25, 'red'), (0.5, 'yellow'), (1, 'cyan')], hours_to_show = [0, 6, 12, 18, 24],ax=None):
    #colors = [(0, 'gray'), (.25, 'cyan'), (0.5, 'yellow'), (1, 'red')]
    
    custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', colors)
    
    xgrid, ygrid = np.meshgrid(np.linspace(0, 24, arr.shape[1] + 1), np.arange(arr.shape[0] + 1))

    if ax: plt.sca(ax)
        
    plt.pcolor(xgrid, ygrid, arr, cmap=custom_cmap, vmin=1, vmax=4)
    plt.colorbar()
    
    for hour in hours_to_show:
        plt.axvline(x=hour, color='k', linestyle='--', linewidth=0.8)
    
    # Set custom x-ticks to include the tick at 24
    plt.xticks(hours_to_show)
    plt.xlabel('Time [h]')
    plt.ylabel('Day index')
    return plt.gca()


def savefig(name, fig_path):
    plt.savefig(fig_path+f'{name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(fig_path+f'{name}.pdf', dpi=300, bbox_inches='tight')


def auc_plot(y_true, y_pred_proba, ax):
    # AUC curve, no uncertainty bars

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculate the AUC
    auc = roc_auc_score(y_true, y_pred_proba)
        
    # Plot the ROC curve
    plt.sca(ax)
    plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.2f)' % auc) #color='blue', 
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')    
    plt.title(f'AUC: {auc:.2f}')

    return auc


def pr_curve_plot(y_true, y_pred_proba, ax):
    # Calculate the Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate the area under the PR curve
    pr_auc = auc(recall, precision)
    
    # Plot the Precision-Recall curve
    plt.sca(ax)
    plt.plot(recall, precision, lw=2, label='PR curve (AUC = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR AUC: {pr_auc:.2f}')
    
    return pr_auc

def plot_segment(ax, t, hours_y, true, predictions, hours_x=None, x=None):
    plt.sca(ax)

    if (hours_x is not None) and (x is not None):
        plt.plot(hours_x[t], x[t], marker='o', c='k', label='Input')
    plt.plot(hours_y[t], true[t], marker='o', c='gray', label='True', lw=0)
    plt.plot(hours_y[t], predictions[t], marker='o', c='r', label='Prediction')

    plt.legend(prop={'size': 8}, loc='upper right', bbox_to_anchor=(1, 1))
    plt.xlabel('Time [h]')
    plt.ylabel('Activity')


def plot_day(ax, day, hours_y_val, days_y_val , y_val, predictions):
    plt.sca(ax)
    # zero = first predicted bin (0-15min)
    mask = days_y_val[:,0] == day

    pred = predictions[mask,0] 
    true = y_val[mask,0]    
    tlin = hours_y_val[mask,0]
    
    plt.scatter(tlin,true, color='gray',s=5)
    plt.scatter(tlin,pred, color='red',s=5)
    plt.title(f'day = {day}')

    plt.legend(['true', 'prediction'], 
               prop={'size': 8},  # Smaller font size
               loc='upper right', # Position the legend
               bbox_to_anchor=(1, 1))  # Keep it within plot bounds    


def plot_auc_for_val(tmin, tmax, ax1, ax2, predictions, y_val, hours_y_val)

    delta = 0 # how many prediction steps, here next 15min
    
    mask = (hours_y_val[:,delta] < tmax) & (hours_y_val[:,delta] > tmin)
        
    true = (y_val[mask,delta] - 1 ) /3
    pred = (predictions[mask, delta] - 1) /3 
    
    # Compute precision-recall pairs and area under precision-recall curve
    precision, recall, _ = precision_recall_curve(true > 0, pred)
    pr_auc = my_auc(recall, precision)
    
    # Compute false positive rate and true positive rate for ROC, and AUC-ROC score
    fpr, tpr, _ = roc_curve(true > 0, pred)
    roc_auc = roc_auc_score(true > 0, pred)
            
    # Plot Precision-Recall Curve
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=ax1)
    ax1.set_ylim([0, 1.05])
    ax1.set_title(f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    
    # Plot ROC Curve
    roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    roc_disp.plot(ax=ax2)
    ax2.set_title(f'ROC curve (AUC = {roc_auc:.2f})')
    
