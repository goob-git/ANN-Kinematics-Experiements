# ANN-Kinematics-Experiements
Constrasting the traditional Inverse Kinematics world, with ANN-based approaches. 

This repository currently consists out of four algorithms that need to be used in a specific order for functionality to be achieved. 

Algorithm 1: Forward Kinematics. 
This algorithm creates random joint angles within specified tolerances, then solves them to create a list of random x, y, z, pitch and joint angle combinations. This is then stored in an Excel file. 

Algorithm 2: Inverse Kinematic Geometric Solver. 
This algorithm reads the Excel file created by the first algorithm, then solves it iteratively to find it's own Inverse Kinematics solutions for the joint angles corresponding to x,y,z position. It stores its solutions as a second sheet. 

Algorithm 3: ANN Inverse Kinematic Solver. 
This algorithm trains a regressor to solve the Inverse Kinematics created by the generation algorithm. It trains off the same spreadsheet, then creates another sheet storing its solutions. 

Algorithm 4: Plotting 
This algorithm is semi-vibecoded currently. Please forgive me time was limited. It just takes the joint angles provided by the two solution algorithms and converts them into a 3-D plot of the robotic arm position. 
