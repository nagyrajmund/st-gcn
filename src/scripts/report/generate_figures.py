import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import csv
import numpy as np

sns.set()

def moving_average(y, N=10):
    y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode='valid')
    return y_smooth

def plot_diagram(x, y, label, title="", show=False, file_name=None):
    plt.plot(x, moving_average(y), label=label)
    matplotlib.rcParams.update({'font.size': 72})
    plt.xlabel("steps", horizontalalignment='right', x=1.0)
    plt.title(title)
    plt.legend()
    fig = plt.gcf()

    if show:
        plt.show()
    if file_name is not None:
        fig.savefig(file_name)

def open_csv(file_name, header=True):
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        if header:
            next(reader, None)
        
        x, y = [], []
        for row in reader:
            x.append(float(row[1]))
            y.append(float(row[2]))
    return x, y

def plot_multiple_diagrams(file_names, labels, title, out_img):
    assert len(file_names) == len(labels)
    nr_of_files = len(file_names)

    for i, (file_name, label) in enumerate(zip(file_names, labels)):
        x, y = open_csv(file_name)
        if i == nr_of_files - 1:
            plot_diagram(x, y, label, title, True, out_img)
        else:
            plot_diagram(x, y, label, title)

def plot_conf_matrix(mat):
    n = mat.shape[0]
    plt.grid(False)
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    for (i, j), z in np.ndenumerate(mat):
        plt.text(j, i, '{:d}'.format(z), ha='center', va='center')
    plt.imshow(mat)
    plt.colorbar()
    fig = plt.gcf()
    return fig