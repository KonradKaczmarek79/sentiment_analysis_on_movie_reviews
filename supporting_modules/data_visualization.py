import matplotlib.pyplot as plt


def count_per_label(labels, names):

    fig_dimensions = len(labels) * 2 + 1
    plt.rcParams['figure.figsize'] = (fig_dimensions, fig_dimensions)
    colors = ('Blue', 'Red', 'Green', 'Yellow', 'Black')
    value_quantity_range = tuple(range(len(labels)))
    for label, name, nr in zip(labels, names, value_quantity_range):
        plt.bar(nr, label, width=0.5, label=name, color=colors[nr])
        plt.legend()

    plt.ylabel('Quantity')
    plt.xlabel('Review type')
    plt.show()
