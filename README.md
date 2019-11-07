Repository to host all software for the Fall 2019 EVolutionary Computation course. 
This project will result in an agent based model of wildfire spread simulating the 2016
Fort McMurry wildfire in Alberta. The project will focus on the data preperation to tune the
stochastic fire spreading decision function with strategies of Genetic Programming and 
symbolic regression. The group members include Maxfield Green, Todd Deluca and Karl Kaiser.  


A sample call to generate synthetic data to test the algorithm on:

python mapsynth.py make_initial_and_final_state data/synth_data/plain/ 20

1st arg: The function used to synthesize the fake data
2nd arg: Time of simulation

Similarly, here is a sample call to begin the genetic evolution

python evolve.py main data/synth_data/plain/init_landscape.npy /data/synth_data/plain/final_landscape.npy 20

1st arg: The function called to begin evolution
2nd arg: The beginning synthetic landscape to build our sim off of
3rd arg: The ending synthetic landscape providing a basis for comparison
4rth arg: The timesteps.
