import os 
from matplotlib import pyplot as plt 


def transform_values(values, fun):
    if fun is not None:
        if fun == 'discretize':
            return list(range(len(values)))
        else:
            return [fun(v) for v in values]
    else: 
        return values


def plot_array(values, x_label='X', y_label='Y', title='', plot_type='r-', transform_f=None, file_name=None):
    if plot_type is None:
        plot_type='r-'

    x = list(range(len(values)))
    y = transform_values(values, transform_f)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x, y, plot_type)

    if file_name is not None:
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        plt.savefig(file_name, dpi=300)
    else:
        plt.show()
    plt.clf()


def plot_zip(zipped_values, x_label='X', y_label='Y', title='', plot_type='r-', transform_x=None, transform_y=None, \
    file_name=None):
    if plot_type is None:
        plot_type='r-'

    x = []
    y = []
    for t in zipped_values:
        x.append(t[0])
        y.append(t[1])
    x = transform_values(x, transform_x)
    y = transform_values(y, transform_y)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x, y, plot_type)

    if file_name is not None:
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        plt.savefig(file_name, dpi=300)
    else:
        plt.show()
    plt.clf()


def plot_zip_two_data(zipped_values, x_label='X', y_labels=('Y1', 'Y2'), titles=('', ''), plot_types=('r-', 'b-'), \
    transform_x=None, transform_ys=(None, None), file_name=None):
    if plot_types is None:
        plot_types=('r-', 'b-')
    if transform_ys is None:
        transform_ys=(None, None)

    x = []
    y0 = []
    y1 = []
    for t in zipped_values:
        x.append(t[0])
        y0.append(t[1][0])
        y1.append(t[1][1])
    x = transform_values(x, transform_x)
    y0 = transform_values(y0, transform_ys[0])
    y1 = transform_values(y1, transform_ys[1])

    fig = plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.xlabel(x_label)
    plt.ylabel(y_labels[0])
    plt.plot(x, y0, plot_types[0])

    plt.subplot(122)
    plt.xlabel(x_label)
    plt.ylabel(y_labels[1])
    plt.plot(x, y1, plot_types[1])

    if file_name is not None:
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        plt.savefig(file_name, dpi=300)
    else:
        plt.show()
    plt.clf()
