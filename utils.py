import random
import numpy as np
import torch
import logging
import os
import sys
import time
import pathlib
import math
import advertorch

# sys.path.append("../../")

def makedirs(dirname):
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)


def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    # write info message
    logger.info('\n\n------ ******* ------ New Log ------ ******* ------')
    return logger


class get_epoch_logger():
    def __init__(self):
        self.epochs = []
        self.results = []
        self.best_epoch = 0; self.best_result = 0
        
    def append_results(self, list):
        self.epochs.append(list[0])
        self.results.append(list[1])
    
    def update_best_epoch(self):
 
        if self.results[-1] >= self.best_result:
            self.best_epoch = self.epochs[-1]
            self.best_result = self.results[-1]
        
        message = 'Best result @ {:03d}, {} \n'.format(self.best_epoch, self.best_result)
        return self.best_epoch, message
    
    def update_best_epoch_to_logger(self, logger):
            
        if self.results[-1] >= self.best_result:
            self.best_epoch = self.epochs[-1]
            self.best_result = self.results[-1]
        
        logger.info('Best result @ {:03d}, {} \n'.format(self.best_epoch, self.best_result))
        return self.best_epoch

class timer():
    def __init__(self):
        # print("Timer initialized @ "+ time.strftime('%Y-%m-%d-%H:%M:%S'))
        self.tic()
        
    def tic(self):
        self.t0 = time.time()
        # print(time.strftime('%Y-%m-%d-%H:%M:%S'))
        
    def toc(self, restart=True):
        diff = time.time()-self.t0
        if restart: self.t0 = time.time()
        return diff


def get_benchmark_sys_info():
    rval = "#\n#\n"
    rval += ("# Automatically generated benchmark report "
             "(screen print of running this file)\n#\n")
    uname = os.uname()
    rval += "# sysname: {}\n".format(uname.sysname)
    rval += "# release: {}\n".format(uname.release)
    rval += "# version: {}\n".format(uname.version)
    rval += "# machine: {}\n".format(uname.machine)
    rval += "# python: {}.{}.{}\n".format(
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro)
    rval += "# torch: {}\n".format(torch.__version__)
    rval += "# advertorch: {}\n".format(advertorch.__version__)
    return rval


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)






import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure as figure
plt.rcParams["font.family"] = "serif"
# plt.rcParams['font.size'] = 12

def log_figure(root, file, data, y_axis=None):
    """
        Docs: plot a figure with data and saved in file
        Args:
            file: pdf prefered
            data: dic with keys including 'x', 'y', 'color', 'title', 'legend'
            
            e.g.: data = {'num_lines':2, 'title':'Activation avg mag',
                    'x_0':x, 'y_0':nat_avg_mag, 'color_0':'red', 'label_0':'nat', \
                    'x_1':x, 'y_1':adv_avg_mag, 'color_1':'blue', 'label_1':'adv',
                    }          
    """
    # pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    
    num_lines = data['num_lines']
    title = data['title']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    width = 0.5
    for i in range(num_lines):
        x = data['x_'+str(i)]
        y = data['y_'+str(i)]
        color = data['color_'+str(i)]
        label = data['label_'+str(i)]
        ax.plot(x, y, '-', color=color, label=label, alpha=0.5/(i+1), linewidth=1)
        ax.fill_between(x, y, np.zeros_like(x), facecolor=color, alpha=0.7/(i+1))
        
        # ax.bar(x, y, width=width, color=color, label=label, alpha=0.7/(i+1))
        
    if y_axis is not None:
        ax.set_ylim(y_axis[0], y_axis[1])
        
    ax.legend(loc='upper right')
    ax.set_title(title)
    fig.savefig(os.path.join(root,file))
    plt.close()
    

def axplot_multi_lines(ax, data_list):
    """
        Docs: plot a figure with data and saved in file
        Args:
            file: pdf prefered
            data: dic with keys including 'x', 'y', 'color', 'title', 'legend'
            
            e.g.: data = [
                {'title':'name','x':x, 'y':nat_avg_mag, 'color':'red', 'label':'nat'},
                ]
    """
    assert isinstance(data_list, list)
    num_lines = len(data_list)
    for i in range(num_lines):
        data = data_list[i]
        title = data['title']
        x = data['x']
        y = data['y']
        color = data['color']
        label = data['label']
        # ax level
        x_label = data['x_label']
        y_label = data['y_label']
        y_axis = data['y_axis']
        linewidth=data['linewidth']
        
        ax.plot(x, y, '-', color=color, alpha=0.8/math.sqrt(i+1), linewidth=linewidth)
        ax.fill_between(x, y, np.zeros_like(x), facecolor=color, alpha=0.8/math.sqrt(i+1), label=label)
        
    if y_axis is not None:
        ax.set_ylim(y_axis[0], y_axis[1])

    ax.legend(loc='upper center', markerscale=1, fancybox=True, fontsize=12)
    ax.margins(x=0.01)
    ax.set_title(title)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    ax.set(facecolor = "ivory")
    ax.grid(axis='x', color='grey', linestyle='--', linewidth=.5)
    ax.grid(axis='y', color='grey', linestyle='--', linewidth=.5)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    

def subplot_figure(root, file, data_plots):
    """
        Docs: 
        Args:
            e.g.: data = [[], []]
    """
    # pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    assert isinstance(data_plots, list)
    num_plots = len(data_plots)
    
    w, h = figure.figaspect(1/2 * num_plots)
    fig = plt.figure(figsize=(w, h))
    
    
    for i in range(num_plots):
        # ax = fig.add_subplot(num_plots, 1, i+1, frameon=False)
        ax = fig.add_subplot(num_plots, 1, i+1, frameon=True)
        axplot_multi_lines(ax=ax, data_list=data_plots[i])    
    
    fig.tight_layout()
    fig.savefig(os.path.join(root,file), dpi=200)
    plt.close()


if __name__ == "__main__":
    import numpy as np
    x = list(range(0,10))
    y = np.random.randn(10)
    data = [
        [{'title':'test1', 'x':x, 'y':y, 'color':'blue', 'label':'hello'},
         {'title':'test1.1', 'x':x, 'y':y+1, 'color':'red', 'label':'hello'}
         ],
        [{'title':'test2', 'x':x, 'y':y, 'color':'blue', 'label':'hello'}],
        ]
    root='.'
    file='test1.png'
    subplot_figure(root, file, data)
