Repository to host all software for the Fall 2019 Evolutionary Computation course. 
This project will result in an agent based model of wildfire spread simulating the 2016
Fort McMurry wildfire in Alberta. The project will focus on the data preperation to tune the
stochastic fire spreading decision function with strategies of Genetic Programming and 
symbolic regression. The group members include Maxfield Green, Todd Deluca and Karl Kaiser.  


A sample call to generate 20 synthetic landscapes, including the layers and the 
initial and final burned landcapes:

```
python mapsynth.py make_synthetic_dataset test_data 20 50 10.1 100 20 --seed 0
```


A sample call to generate initial and final burn landscapes from 
synthetic layer files:

```
python mapsynth.py make_initial_and_final_state data/synth_data/plain/inputs 20
```

1st arg: The function used to synthesize the fake data
2nd arg: Time of simulation

Similarly, here is a sample call to begin the genetic evolution:

```
python evolve.py main data/synth_data/plain/inputs/init_landscape.npy data/synth_data/plain/inputs/final_landscape.npy 20
```

1st arg: The function called to begin evolution
2nd arg: The beginning synthetic landscape to build our sim off of
3rd arg: The ending synthetic landscape providing a basis for comparison
4rth arg: The timesteps.

## Wildfire Simulation

At every time step, simulation proceeds as follows:

1. Get every cell bordering the fire. A cell is on the border of the fire if it's cell 
state is `tree` (and it is not in a `fire` state, which is the case for mutually exclusive states),
and it is adjacent to a cell with a `fire` state.

2. For each cell bordering the fire, get the neighborhood of the cell and call the fire probability function
passing in all the landscape values in the neighborhood.

3. For each border cell, randomly sample from the uniform distribution. If the sample is less than the fire
probability for that cell, change the cell's state to `fire` from `tree`.
