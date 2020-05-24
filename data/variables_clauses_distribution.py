import os 
import pandas as pd
from matplotlib import pyplot as plt 

variables_dist = []
clauses_dist = []
this_dirname = os.path.dirname(__file__)
cnf_dirname = os.path.join(this_dirname, 'cnf')

for _, _, files in os.walk(cnf_dirname):
    for file in files:
        with open(os.path.join(cnf_dirname, file)) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('p cnf'):
                    tokens = line.strip().split(' ')
                    variables = int(tokens[2])
                    clauses = int(tokens[3])
                    variables_dist.append(variables)
                    clauses_dist.append(clauses)

variables_dist.sort()
clauses_dist.sort()
num_of_cnfs = len(variables_dist)

fig_dirname = os.path.join(this_dirname, 'metadata', 'figures', 'distributions')
if not os.path.exists(fig_dirname):
    os.makedirs(fig_dirname)

figname = os.path.join(fig_dirname, 'variables_clauses_distribution.png')
fig = plt.figure(figsize=(12, 6))

plt.subplot(121)
x = list(range(num_of_cnfs))
y = variables_dist
plt.xlabel('Instance id (sorted)')
plt.ylabel('Number of variables')
plt.plot(x, y, 'r-')

plt.subplot(122)
x = list(range(num_of_cnfs))
y = clauses_dist
plt.xlabel('Instance id (sorted)')
plt.ylabel('Number of clauses')
plt.plot(x, y, 'b-')

plt.savefig(figname)
plt.clf()