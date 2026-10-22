import numpy as np
import pandas as pd
import time


#startTime1 = time.perf_counter()
rng = np.random.default_rng()
pitchList = []
dataframe = 'adataset.xlsx'
theta = 0
numSamples = 50  #amount of samples to gen
jointLimits = [[-180, 180], [-90, 90], [0, 90], [-90, 90]] #jointlims for physical arm

dhparams = [ #dh parameters for arm
    [30, 0, np.pi / 2], #J1 
    [0, 195, 0		 ], #J2
    [0, 195, 0       ], #J3
    [0, 94, 0        ]  #J4
    ]

def generateRandomPitches(): #depricated function
    desiredPitch = rng.uniform(-90, 90)
    return desiredPitch


def createDHMatrix(i, radians): #sets up a dh matrix based on DH parameters and joint angles passed
    d_val, theta_val, a_val, alpha_val  = dhparams[i][0], radians, dhparams[i][1], dhparams[i][2]
    
    #simpler functions for sin and cos
    cosT, sinT = np.cos(theta_val), np.sin(theta_val) 
    cosA, sinA = np.cos(alpha_val), np.sin(alpha_val)
    
    return np.array([
        [cosT, -sinT * cosA,  sinT * sinA, a_val * cosT],
        [sinT,  cosT * cosA, -cosT * sinA, a_val * sinT],
        [   0,         sinA,         cosA,        d_val],
        [   0,            0,            0,            1]
    ])

def createRandomJointAngles(numSamples, jointLimits): #creates random joint angles between physical tolerances
    jointRadiansList = []
    while len(jointRadiansList) < numSamples:
        J1 = np.radians(rng.uniform(jointLimits[0][0], jointLimits[0][1])) #uses numpy uniform and joint limits
        J2 = np.radians(rng.uniform(jointLimits[1][0], jointLimits[1][1]))
        J3 = np.radians(rng.uniform(jointLimits[2][0], jointLimits[2][1]))
        J4 = np.radians(rng.uniform(jointLimits[3][0], jointLimits[3][1]))
        
        pitch = J2 + J3 + J4 #computes global pitch 
        
        jointRadiansList.append([J1, J2, J3, J4]) #stores joint angles
        pitchList.append(pitch) #stores pitch
            
    return jointRadiansList

def matrixFeeder(jointRadiansList): #creates DH matrixes for each full movement
    
    bigMatrixList = [] #stores matrixes for entire dataset
    
    for joints in jointRadiansList:
        matrices = [] #stores 4 matrixes for 4 joints in this record
        for i, radians in enumerate(joints):
            currentMatrix = createDHMatrix(i, radians) #create DH matrix 
            matrices.append(currentMatrix)  #store it in local list    
        bigMatrixList.append(matrices) #store list of 4 in global list
            
    return bigMatrixList

def FKCalculator(matrixList): #calculates forward kinematics x,y,z based on joint angles
    calculatedCoords = []
    
    for matrixes in matrixList:
        result = np.linalg.multi_dot(matrixes) #literlly just multiply the 4 matrixes
        
        #strip the x, y, z coordinates from them 
        x = result[0][3]
        y = result[1][3]
        z = result[2][3]
        calculatedCoords.append([x, y, z])
   
    return calculatedCoords #return coordinates found
    
def dataFrameMaker(jointRadiansList, coordinates, pitchList, dataframe): #makes pretty spreadsheet of joints mapped to coordinates. 
    
    radiansdf = pd.DataFrame(jointRadiansList, columns = ['joint1_FK', 'joint2_FK', 'joint3_FK', 'joint4_FK']) #VERY IMPORTANT! Joint angles are radians not degrees!!
    coordinatesdf = pd.DataFrame(coordinates, columns = ['x', 'y', 'z'])
    pitchdf = pd.DataFrame(pitchList, columns = ['pitch'])
    finaldf = pd.concat([radiansdf, coordinatesdf, pitchdf], axis = 1)
    finaldf.to_excel(dataframe, sheet_name='ProposedCoordinatesAngles', index=True) 
    
    return finaldf #return dataframe created in case want to display or use further

print("================================================================================\n [1/1] CREATING LIST OF RANDOM JOINT ANGLES, SOLVING WITH FORWARD KINEMATICS \n================================================================================")
jointRadiansList = createRandomJointAngles(numSamples, jointLimits)
print('Creating DH Parameter Matrixes')
matrixes = matrixFeeder(jointRadiansList)
print("Computing Forward Kinematics For Created Matrix List")
coordinates = FKCalculator(matrixes)
print("Storing Resultant Coordinate-Angle Sets in Dataframe")
dataset = dataFrameMaker(jointRadiansList, coordinates, pitchList, dataframe)
print('\nCompleted Successfully,  Have a Nice Day!')

#for i, matrix in enumerate(matrixes):
    #print("Matrixes for angle set \n ", i, " : ", matrix, '\n')
#endTime1 = time.perf_counter()
#executionTime1 = endTime1 - startTime1
#print("Time taken to generate samples:\n", executionTime1)
