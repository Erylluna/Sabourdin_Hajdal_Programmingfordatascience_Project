import Matrixclass as M
import Graphsfortimecomparison as Graph
import os
import cProfile
import sys

def profiler_all_matrices_functions():
    '''Function that uses the cprofile method on Graph.measure_time_operation (100) and displays it on snakeviz to show the programmer what aspects of the programm could be optimized'''
    #Do the Cprofile analysis
    cProfile.run('Graph.measure_time_operations(100)', 'measure_time_operations_profile.cprof')
    #Graph.measure_time_operations(100) was chosen to run the profiler as it runs all the methods of the Matrix class
    #The value 100 was chosen arbitrarly, as it is the highest value for which my computer can run the methods in an acceptable time
    
    #Open the analysis in snakeviz to have a clear view of the data
    os.system(f"snakeviz measure_time_operations_profile.cprof")
