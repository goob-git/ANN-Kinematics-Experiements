import numpy as np
import pandas as pd
import psutil as util
import time as time
import os
#hardware tracking
process = util.Process(os.getpid())

#this algorithm works very well assuming we J2 has the limits 0, 90.
#assuming millimeters offset from base
baseOffset = 30
elOffset23 = 195
elOffset4 = 94
dataframe = 'smalldataset.xlsx'

#counter variables
distanceViolation = 0
largeAngleViolation = 0
smallAngleViolation = 0


def pitchSolver(J2, J3, J4): #only needed when working from dataset excluding pitch like first generation function
    pitch = J2 + J3 + J4
    return pitch

def baseSolver(x, y, z):
    baseR = np.arctan2(y, x) #angle of J1 needed to be pointing in correct direction
    targetXY = np.sqrt((x**2 + y**2)) #xy distance to target from base
    vertOffset = z - baseOffset #shoulder is now base, offset accounted for
    
    return baseR, targetXY, vertOffset

def calculateWristPos(targetXY, vertOffset, pitch, elOffset4):
    xWrist = targetXY - elOffset4 * np.cos(pitch) #we get the horizontal offset of J4 from EF
    zWrist = vertOffset - elOffset4 * np.sin(pitch) #we get the veritcal offset of J4 from EF
    J2J4Dist = np.sqrt(xWrist ** 2 + zWrist ** 2) #we get the straightline distance from J2 to J4
    
    return xWrist, zWrist, J2J4Dist

def applyCosLaw(J2J4Dist):
    
    #c^2 = a^2 + b^2 - 2abcos(')
    cosLaw = ((2 * (elOffset23 ** 2)) - J2J4Dist**2) / (2 * (elOffset23 ** 2)) #apply law of cosines to find internal angle J3
    cosLaw = np.clip(cosLaw, -1, 1) #floating point errors
    intAngle = np.arccos(cosLaw) #extract internal angle radians 0 - 180
    
    return intAngle

def calculatePhi(zWrist, xWrist):
    phi = np.arctan2(zWrist, xWrist) #find global elevation of vector from J2 - J4, preserves CAST sector
    
    return phi
    
def calculateTheta(cosLawElbow):
    theta = ((np.pi - cosLawElbow) / 2) #how many radians tilt away from straight link J2 - J4
    
    return theta
    
def calculateJ2(phi, theta):
    
    #we assume elbow up configuration for this arm. Elbow down is possible but limited use case, can be implemeted by inverting.
    #cool idea is evaluating both then comparing closest to current angle for smoother motion
    J2ElbowUp = phi - theta #elbow up configuration
    
    return J2ElbowUp

    
def calculateJ3(intAngle):
    
    J3 = np.pi - intAngle #supplementary angle to internal angle
    
    return J3
    
def calculateJ4(J2, J3, pitch):
    
    J4 = pitch - (J2 + J3) #remove J4 from global pitch.
    
    return J4


def inverseKinematics(x, y, z, pitch):
    
    global distanceViolation, largeAngleViolation, smallAngleViolation #scope counter variables
    
    J1, targetXY, vertOffset = baseSolver(x, y, z) 
    xWrist, zWrist, J2J4Dist = calculateWristPos(targetXY, vertOffset, pitch, elOffset4)

    if J2J4Dist > (2 * elOffset23) or J2J4Dist < 0: #check if desired position physically reachable
        distanceViolation += 1
        return None
    
    #callinig functions to calculate internal angle, phi, theta and then J2 - J4 using values
    intAngle = applyCosLaw(J2J4Dist) 
    phi = calculatePhi(zWrist, xWrist)
    theta = calculateTheta(intAngle)

    J2= calculateJ2(phi, theta)
    
    J3 = calculateJ3(intAngle)

    J4 = calculateJ4(J2, J3, pitch)
    
    
    if J2 > np.pi/ 2 or J3 > np.pi or J4 > np.pi / 2: #check if angle needed too large
        largeAngleViolation += 1
        return None
    
    if J2 < -np.pi / 2 or J3 < -np.pi /2 or J4 < -np.pi/ 2: #check if angle needed too small
        smallAngleViolation += 1
        return None

    J1Deg, J2Deg, J3Deg, J4Deg = np.degrees(J1), np.degrees(J2), np.degrees(J3), np.degrees(J4) #convert radians into degrees
    
    radsDegreesList = [J1, J2, J3, J4, J1Deg, J2Deg, J3Deg, J4Deg] #returns radians J1 - J4, and degrees J1 - J4
    
    return radsDegreesList

def readFromDataframe(dataframe): #load dataframe into pandas, then numpy array
    
    print('\n===================================================\n [1/3] LOADING DATAFRAME INTO MEMORY FOR COMPUTATION\n===================================================')
    originalDF = pd.read_excel(dataframe, usecols = ['x', 'y', 'z', 'pitch'])
    print('Dataframe Load Success, First Five Records Displayed As Proof')
    print(originalDF.head())
    
    return originalDF.to_numpy()
    
def writeToDataframe(calculatedRadsDegrees, dataframe): #makes pretty sheet in excel with joint angles in degrees and radians. 
    
    calculatedDF = pd.DataFrame(calculatedRadsDegrees, columns = ['joint1_predicted_r', 'joint2_predicted_r', 'joint3_predicted_r', 'joint4_predicted_r', 'joint1_predicted_d', 'joint2_predicted_d', 'joint3_predicted_d', 'joint4_predicted_d']) #VERY IMPORTANT! Joint angles are radians not degrees!!
    with pd.ExcelWriter(dataframe, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        calculatedDF.to_excel(writer, sheet_name='SolvedAnglesRadsDegrees', index=True)

#main loop
originalNP = readFromDataframe(dataframe) #load dataframe
bigList = [] #store results
resourceLogs = []  #store resource usage
 
#log CPU, memory and time 
process.cpu_percent(interval = None)
initialMem = process.memory_info().rss / (1024**2) #MB
startTime = time.perf_counter()

for row in originalNP: #iterate through unsolved dataset, taking x, y, z, pitch
    
    CalculatedRadsDegrees = inverseKinematics(row[0], row[1], row[2], row[3]) #calculate angles required using IK
    
    if CalculatedRadsDegrees != None: #if valid solution found, store it
        bigList.append(CalculatedRadsDegrees)
    else: #if valid solution not found, fill row with none values
        bigList.append(['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'])
        
writeToDataframe(bigList, dataframe) #store it as a second sheet on original excel document
#output solver statistics
print('\n===================================================\n [2/3] STARTING INVERSE KINEMATICS ALGORITHM\n===================================================')
print('Physical Reach Violations Left Unsolved: ', distanceViolation)
print('Solutions Requiring Smaller Joint Angles Than Physically Reachable: ', smallAngleViolation)
print('Solutions Requiring Larger Joint Angles Than Physically Reachable: ', largeAngleViolation)

endTime = time.perf_counter()
cpuUtil = process.cpu_percent(interval = None)
cpuUtiltot = (process.cpu_percent(interval = None) / util.cpu_count())
postMem = process.memory_info().rss / (1024**2)

#output runtime statistics
print('\n===================================================\n [3/3] OUTPUTTING HARDWARE AND RUNTIME STATS AFTER COMPUTATION \n===================================================')
print('Total Dataset Record Evaluated: ', originalNP.size)
print("Total Memory Usage in Megabytes: ", postMem - initialMem)
print("Average Single Core Usage Over Algorithm Runtime: ", cpuUtil)
print("Total Calculation Runtime: ", endTime - startTime, "to execute.")
print("\nData Appended Successfully :)")

    




    