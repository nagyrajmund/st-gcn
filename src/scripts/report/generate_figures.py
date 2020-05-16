import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np

sns.set()

def plot_diagram(x, y, label, title="", show=False, file_name=None):
    plt.plot(x, y, label=label)
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

# x, y = open_csv("logs/run-version_1-tag-loss.csv")
# plot_diagram(x, y, "First", "", False)
# x, y = open_csv("logs/run-version_2-tag-loss.csv")
# plot_diagram(x, y, "Second", "", False)
# x, y = open_csv("logs/run-version_3-tag-loss.csv")
# plot_diagram(x, y, "Third", "Title", True, "img/test.png")

# file_names = ["logs/run-version_1-tag-loss.csv", "logs/run-version_2-tag-loss.csv", "logs/run-version_3-tag-loss.csv"]
# labels = ["First", "Second", "Third"]
# plot_multiple_diagrams(file_names, labels, "Title", "img/test.png")