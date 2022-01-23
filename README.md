# paper-multicomponent-evolution
Code for the paper "Evolved interactions stabilize many coexisting phases in multicomponent fluids".

All the code is contained in the python module `multicomponent_evolution.py`.
The only required python modules are `numpy`, `scipy`, and `numba`, which are listed in the `requirements.txt`.

The modules contains a few global constants, which set parameters of the algorithm as described in the paper.
They typically do not need to be changed.
A good entry point into the code might be to create a random interaction matrix and a random initial composition using `random_interaction_matrix` and `get_uniform_random_composition`, respectively.
The function `evolve_dynamics` can then be used to evolve Eq. 4 in the paper to its stationary state, whose composition the function returns.
The returned composition matrix can be fed into `count_phases` to obtain the number of distinct phases.
An ensemble average over initial conditions is demonstrated in the function `estimate_performance`, which also uses Eq. 5 of the paper to estimate how well the particular interaction matrix obtains a given target number of phases.
Finally, `run_evolution` demonstrates the evolutionary optimization over multiple generations.