import matplotlib.pyplot as plt
import seaborn as sns
import csv

sns.set()

def plot_diagram(x, y, label, show=False, file_name=None):
    plt.plot(x, y, label=label)
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

def plot_multiple_diagrams(file_names, labels, out_img):
    assert len(file_names) == len(labels)
    nr_of_files = len(file_names)

    logs = list(zip(file_names, labels))

    for i in range(nr_of_files):
        show = (i == nr_of_files - 1)
        file_name, label = logs[i]
        x, y = open_csv(file_name)
        plot_diagram(x, y, label, show)

# x, y = open_csv("logs/run-version_1-tag-loss.csv")
# plot_diagram(x, y, "First", False)
# x, y = open_csv("logs/run-version_2-tag-loss.csv")
# plot_diagram(x, y, "Second", False)
# x, y = open_csv("logs/run-version_3-tag-loss.csv")
# plot_diagram(x, y, "Third", True, "img/test.png")

file_names = ["logs/run-version_1-tag-loss.csv", "logs/run-version_2-tag-loss.csv", "logs/run-version_3-tag-loss.csv"]
labels = ["First", "Second", "Third"]
plot_multiple_diagrams(file_names, labels, "img/test.png")