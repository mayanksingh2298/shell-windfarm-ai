# Tools created by Mayank

## Helpful scripts
1. To display the heatmap of a configuration:  
`python3 heatmap.py <path to csv>`

2. To visualize the output of a configuration:  
`python3 visualize_output.py <path to csv>`

3. To generate the windrose diagram of a wind distribution:  
`python3 windrose_gen.py <path to wind distribution>`

## Algorithms implemented

1. 5-var: Uses only 5 variables to represent all the 50 turbines based on the geometry. `x-seperation`, `y-separation`, `theta` to rotate everything by, `x-offset` and `y-offset`. Does a simple localsearch -> changing 1 variable at a time and if the score improves, keeps the new placement as the current placement.
```
cd 5-var
python3 main.py
```

2. Tried using different evolutionary algos like `pso`, `pdfo`, `cmaes`. CMAES seemed to work the best. The idea is that using simple local search techniques first and then when its solution saturates, give it to cmaes it would give a new solution, now run local search again.
```
cd evolutionary_algos
python3 main-cma.py <path to initial placement>
python3 main-pso.py <path to initial placement>
python3 main-pdfo.py <path to initial placement>
```

3. Wrote a parallel architecture which spawns multiple processes and uses simple localsearch, localsearch with negative step and rajas's local search. It works as follows: There is a master process which reads data/winner.csv and sends it to all children which run their local algorithms and keep updateing data/children/<child>.csv. The master in a round robin fashion looks at all these children files and if the best of them is better than data/winner.csv, it replaces that as the new winner. There are multiple hyperparameters which can be tuned like:
	1. NPROCS - total number of children to spawn
	2. NEGATIVE_CHILDREN - the number of local search with negative step processes
	3. RAJAS_CHILDREN - the number of rajas's local search processes
	4. CHILDREN_IMPROVEMENT_THRESHOLD - child process will update its local file if it sees atleast this much improvement
	5. MASTER_IMPROVEMENT_THRESHOLD - master process will update the global winner if it sees atleast this much improvement
```
python3 main.py
```