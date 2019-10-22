Repository to host all software for the Fall 2019 EVolutionary Computation course. 
This project will result in an agent based model of wildfire spread simulating the 2016
Fort McMurry wildfire in Alberta. The project will focus on the data preperation to tune the
stochastic fire spreading decision function with strategies of Genetic Programming and 
symbolic regression. The group members include Maxfield Green, Caroline Popiel, Todd Deluca and Karl.  


A sample call to a basic simulation:

python spread_fire.py data/perlin_50_x_50_100.npy 20

1st arg: landscape raster in form of numpy array
2nd arg:t  time of simulation

Ouput : t images placed in the fire_gif folder. Use image magick to animate results
