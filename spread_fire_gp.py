#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:55:23 2019

@author: max
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:06:26 2018
@author: mgreen1
0 = firebreak, 1 = fire, 2 = tree
"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib.colors as colors
import sys
import os

class Cell():
    # constructor for cell
    def __init__(self,x,y,z,state):
        self.x = x
        self.y = y
        self.position = x,y
        self.z = z
        self.state = state
        self.visited = False
        self.risk = 0
        self.fireT = 0
        self.partition = -1
        self.dz1 = None
        self.dz2 = None
        self.dz3 = None
        self.dz4 = None
        self.nStates = 2
        self.p = []
        self.maxDz = 0
        self.spatial_risk = None
        
    def getNTFire(self,landscape):
        """Get fire times of all neighbors
        """
        neighbor_fire_times = []
        i,j = self.getPosition()
        for n in self.getN(landscape):
            neighbor_fire_times.append(landscape[n].fireT)
        return(neighbor_fire_times)
    

    def getNState(self,landscape):
        """
        Return states of all neighboring cells
        
        """
        i = self.x
        j = self.y
        try:
            n1 = landscape[i-1,j].getState()
        except:
            IndexError
        try:
            n2 = landscape[i,j+1].getState()
        except:
            IndexError
        try:
            n3 = landscape[i+1,j].getState()
        except:
            IndexError
        try:
            n4 = landscape[i,j-1].getState()
        except:
            IndexError

        # Build case for each border area, upper/lower left corner, upper/lower right corner
        # All Four borders
        # Upper Left Corner (No n1 or n4)
        if i == 0 and j == 0:
            return(n2,n3)
        # Upper right corner(no n1 or n2)
        elif i==0 and j==len(landscape)-1:
            return(n3,n4)
        # Lower left corner(no n3,n4)
        elif i == len(landscape)-1 and j == 0:
           return(n1,n2)
        # Lower right corner(no n2 or n3)
        elif i == (len(landscape)-1) and j == (len(landscape)-1):
            return(n1,n4)
        # On top of matrix
        elif i ==0:
            return(n2,n3,n4)
        # Bottom of matrix
        elif i == len(landscape)-1:
            return(n1,n2,n4)
        # Right side of matrix
        elif j == len(landscape)-1:
            return(n1,n3,n4)
        # Left Side of matrix
        elif j == 0:
            return(n1,n2,n3)
        else:
            return(n1,n2,n3,n4)

    def getN(self,landscape):
        """
        Return neighbors of cell account for all edge cases, fire does not spread on torus
        """
        i,j = self.getPosition()
        # TRY EXCEPT BLOCK TO ATTEMPT TO ASSIGN NEIGHBOR LOCATIONS
        try:
            n1 = landscape[i-1,j].getPosition()
        except:
            IndexError
        try:
            n2 = landscape[i,j+1].getPosition()
        except:
            IndexError
        try:
            n3 = landscape[i+1,j].getPosition()
        except:
            IndexError
        try:
            n4 = landscape[i,j-1].getPosition()
        except:
            IndexError
            # Build case for each border area, upper/lower left corner, upper/lower right corner
        # All Four borders

        # Upper Left Corner (No n1 or n4)
        if i == 0 and j == 0:
            return(n2,n3)
        # Upper right corner(no n1 or n2)
        elif i==0 and j==len(landscape)-1:
            return(n3,n4)
        # Lower left corner(no n3,n4)
        elif i == len(landscape)-1 and j == 0:
           return(n1,n2)
        # Lower right corner(no n2 or n3)
        elif i == (len(landscape)-1) and j == (len(landscape)-1):
            return(n1,n4)
        # On top of matrix
        elif i ==0:
            return(n2,n3,n4)
        # Bottom of matrix
        elif i == len(landscape)-1:
            return(n1,n2,n4)
        # Right side of matrix
        elif j == len(landscape)-1:
            return(n1,n3,n4)
        # Left Side of matrix
        elif j == 0:
            return(n1,n2,n3)
        else:
            return(n1,n2,n3,n4)

    # getter for state of cell
    def getState(self):
        return self.state

    #setter for state of cell
    def setState(self,state):
        self.state = state

    # Get position of cell in matrix
    def getPosition(self):
        return(self.x,self.y)

    # Get height of cell
    def getZ(self):
        return(self.z)
        
    # Set dz values between site and neighbouring nodes
    def setDz(self,landscape):
        #INITIALIZD DELZ AS NONE
        self.dz2 = None
        self.dz4 = None
        self.dz1 = None
        self.dz3 = None
        for i in range(len(landscape)):
            for j in range(len(landscape)):
        # Exception for higher borders of grid
                try:
                    self.dz1 = landscape[i,j].getZ() - landscape[i+1,j].getZ()
                    self.dz3 = landscape[i,j].getZ() - landscape[i,j+1].getZ()
                except:
                    IndexError
                # Exception for lower borders of grid
                if i!= 0:
                    self.dz2 = landscape[i,j].getZ() - landscape[i-1,j].getZ()
                if j!= 0:
                    self.dz4 = landscape[i,j].getZ() - landscape[i,j-1].getZ()

    def getDz(self):
        return(self.dz1,self.dz2,self.dz3,self.dz4)

    def getDzSum(self,landscape):
        """
        Get sum of height differences from neighbor of current fire cell
        """
        nbs = self.getN(landscape)
        zs = []
        for n in nbs:
            if landscape[n].state == 1:
                zs.append(landscape[n].getZ()-self.z)
        sumDz= np.sum(zs)
        zs.extend([1])
        self.dzMax = np.max(zs)
        return(sumDz)
        
        
    
        
                    
def getStates(landscape):
    """ RETURN ARRAY WITH THE STATE OF EVERY SITE IN LANDSCAPE"""
    state_map = np.zeros([len(landscape), len(landscape)])
    for i in landscape:
        for j in i:
            state_map[j.position] = j.state
    return(state_map)


def max_risk_pos(landscape, potential_fire_sites,place):
    """
        MAX_RISK_POS: calculates the riskiest site for the agent to move to
    """
    #store a list of risks:
    risks = []
    spatial_risks = []
    potential_fire_sites = list(potential_fire_sites)
    #get the risk values for the potential fire sites:
    for site in potential_fire_sites:
        risks.append(landscape[site].risk)
        spatial_risks.append(landscape[site].spatial_risk)
    
    #get the coordinate for the most risky site:
    if place == True:
        riskiest = potential_fire_sites[np.argmax(spatial_risks)]
    else:
        riskiest = potential_fire_sites[np.argmax(risks)]
    
    #return the riskiest site:
    return(riskiest)
    
def get_fire_sites(landscape):
    """
    Retrieve list of sites that are on fire
    
    """
    fire_sites = []
    for i in range(len(landscape)):
        for j in range(len(landscape)):
            if landscape[i,j].getState() == 1:
                fire_sites.append((i,j))
    return(fire_sites)
    


# THIS IS WHAT WE WILL NEED TO CHANGE FOR THE EC PROJECT! 
def update_p_fire(landscape,gamma,zMax):
    """
    UPDATE RISK OF EVERY CELL IN THE LANDSCAPE 
    """
    for i in landscape:
        for j in i:
            # ONLY UPDATE IF CELL IS A TREE
            if j.state == 2:
                # GET STATES OF BORDERS SITES NEIGHBORS
                nStates = j.getNState(landscape)
                # GET SUM OF DELTA Z 
                dzSum = j.getDzSum(landscape)
                nF = Counter(nStates)
                nF = nF[1]
                nS = len(nStates)
                # ASSIGN RISK
                if dzSum == 0:
                    dzSum =1
                # TODO FIX THIS!!!!! PROBABILITY FUNCTION WILL BE TUNED FROM GP
                j.risk = gamma + (1-gamma)*(dzSum)/(nS*2)#j.maxDz)
            # IF CELL IS ALREADY ON FIRE, RISK IS ZERO
            else:
                j.risk = 0
                
                
def fire_init(landscape,gamma, zMax,threshold,init_time):
    """
    Description:
        Main function that progresses simulation over time.
    
    Inputs:
        landscape : Matrix of heights of cells in space
        gamma : value between 0,1 that serves as a weight for the importance of height
        zMax : maximum height in current lansdcape
        threshold: time for fire cells to expire
        init_time: amount of time to run the simulation 
    Outputs:
        stateMaps : a state matrix at each time step from the simulation
    
    sample call:
        fire_init(landscape,8,5,20)
     """
     
    stateMaps = []
    fired = []
    # START FIRE IN CENTER OF LANDSACPE
    i = int(len(landscape)/2)
    j = int(len(landscape)/2)
    # SET STATE OF CELL TO FIRE
    landscape[i,j].setState(1)
    # ADD TO LIST OF FIRED CELLS
    fired.append((i,j))
    t = 0
    # BEGIN FIRE PROPOGATION
    while t < init_time:
        border = []
        # CREATE FIRE BORDER BY VISTING FIRE CELLS THAT ARE NEIGHBORS WITH TREES
        for site in fired:
            # LOOP OVER LIST OF NEIGHBORS OF FIRE CELLS
            for idxN,neighbor in enumerate(landscape[site].getN(landscape)):
                # IF CELL HAS A NEIGHBOR THAT IS A TREE, ADD TREE CELL TO BORDER
                if landscape[neighbor].state == 2:
                    border.append(neighbor)
                    # TURN OLD FIRES INTO ASH/FIREBREAKS
            #if landscape[site].fireT == threshold:
             #   landscape[site].setState(0)
                # KEEP TRACK OF TIME THAT FIRE HAS BEEN BURNING AT A SITE
            landscape[site].fireT += 1
        # CONSIDER ALL BORDER SITES FOR POTENTIAL FIRE SPREADING
        for site in border:
            # DETERMINE PROBABILITY OF FIRE SPREAD
            probFire = landscape[site].risk
            # SET FIRE DEPENDING ON LIKELYHOOD
            if probFire > np.random.rand():
                landscape[site].setState(1)
                fired.append(site)
        t = t+1
        # UPDATE RISK VALUES FOR ALL CELLS IN LANDSCAPE
        update_p_fire(landscape,gamma,zMax)
        stateMaps.append(getStates(landscape))
    return(stateMaps)              
    



# TEST CASE ON PERLIN NOISE LANDSCAPE
landscape_file = sys.argv[1]
time_steps = int(sys.argv[2])


bowlSmall = np.load(landscape_file)
# initialize contained
contained = False

zVals = bowlSmall
N = len(zVals)
landscape = np.ndarray([N,N],dtype = Cell)
for i,ik in enumerate(zVals):
    for j,jk in enumerate(ik):
        z = zVals[i,j]
        a = Cell(i,j,z,2)
        landscape[i,j] = a

# SET HEIGHTS OF CELLS
for i in list(range(len(landscape))):
            for j in list(range(len(landscape))):
                landscape[i][j].setDz(landscape)
                
#initialize fire cluster, determine how many time steps we want the fire to run 
                
stateMaps = fire_init(landscape,.7,4,5,time_steps)
#propogate fire


try:
    os.mkdir("gif_fire")
except:
    FileExistsError
    
for i,frame in enumerate(stateMaps):
    fig, ax = plt.subplots(figsize=(15, 10))
    cmap = colors.ListedColormap(['red', 'green'])
    cax = ax.matshow(frame,cmap = cmap)
    plt.contour(zVals, colors = "b")
    figname = "gif_fire/{}.png".format(i)

    plt.savefig(figname)
    plt.close(fig)

