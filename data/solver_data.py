import os 
import pandas as pd
from matplotlib import pyplot as plt 

this_dirname = os.path.dirname(__file__)

def calculate_solver_data_for_csv(csv_file):
    solvers_data = {}

    csv_file = os.path.join(this_dirname, 'metadata', csv_file)
    data = pd.read_csv(csv_file)
    n = data.count()[0]

    for i in range(n):
        solver_name = data.iloc[i]['solver']
        solver_data = pd.Series([data.iloc[i]['benchmark'], data.iloc[i]['solver time']])
        if solver_name not in solvers_data:
            solvers_data[solver_name] = [solver_data]
        else:
            solvers_data[solver_name].append(solver_data)

    return solvers_data


def create_a_figure_for_a_solver(figures_dirname, name, data, number):
    number = str(number)
    if len(number) == 1:
        number = '0' + number
    figname = os.path.join(figures_dirname, number + name + '.png')

    x = list(range(len(data)))
    y = [series[1] for series in data]
    plt.xlabel('Instances')
    plt.ylabel('Solver time')
    plt.title('Data for solver: ' + name)
    plt.plot(x, y, 'ro')
    plt.savefig(figname)
    plt.clf()


def create_figures_for_solver_data(solvers_data, csv_file):
    csv_figure_dirname = csv_file
    ext_idx = csv_figure_dirname.rfind('.csv')
    if ext_idx != -1:
        csv_figure_dirname = csv_figure_dirname[:ext_idx]
    figures_dirname = os.path.join(this_dirname, 'metadata', 'figures', csv_figure_dirname)
    if not os.path.exists(figures_dirname):
        os.makedirs(figures_dirname)

    solver_number = 0
    for solver_name in solvers_data:
        data = solvers_data[solver_name]
        create_a_figure_for_a_solver(figures_dirname, solver_name, data, solver_number)
        solver_number += 1


main_data = calculate_solver_data_for_csv('main_extracted.csv')
create_figures_for_solver_data(main_data, 'main_extracted.csv')

random_data = calculate_solver_data_for_csv('random_extracted.csv')
create_figures_for_solver_data(random_data, 'random_extracted.csv')