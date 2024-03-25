import Matrixclass as Matrix
import Graphsfortimecomparison as Graph
import numpy as np
import sys
import profiler
import module21

def main():
    '''The main function iterates tests within sys.argv to determine what exactly the user wants out of the program, and perform the required action by calling the different modules'''
    #Checks if the user requires a function in part 1
    if sys.argv[1] == 'part1':
        print('-----PART 1-----')

        #Checks if the user runs the part Operations
        if sys.argv[2].lower() == 'operations':
            print('Now running: matrix operations demonstration')
            
            #define non-singular square matrices 
            matrixa = Graph.create_nonsingular__dense_squarematrix(int(sys.argv[3]))
            matrixb = Graph.create_nonsingular__dense_squarematrix(int(sys.argv[3]))
            matrix1 = Graph.create_nonsingular__sparse_squarematrix(int(sys.argv[3]))
            matrix2 = Graph.create_nonsingular__sparse_squarematrix(int(sys.argv[3]))
        
            # Test Dense addition
            print("Dense Addition:")
            result_add = matrixa.Dense_add(matrixb)
            if result_add:
                print(result_add.densematrix)
            print()

            # Test Dense subtraction
            print("Dense Subtraction:")
            result_subtract = matrixa.Dense_subtract(matrixb)
            if result_subtract:
                print(result_subtract.densematrix)
            print()

            # Test Dense multiplication
            print("Dense Multiplication:")
            result_multiply = matrixa.Dense_multiply(matrixb)
            if result_multiply:
                print(result_multiply.densematrix)
            print()
            
            # Test the todense method
            print("Dense representation of matrices:")
            print(matrix1.todense())
            print(matrix2.todense())
            print()

            # Test the getsparsity method
            print("Sparsity of matrix1:", matrix1.getsparsity())
            print()

            # Test matrix multiplication
            print("Result of matrix multiplication:")
            if matrix1.Sparse_multiply(matrix2):
                print(matrix1.Sparse_multiply(matrix2).CSCmatrix)
            print(np.dot(np.array(matrix1.todense()), np.array(matrix2.todense())))
            print()

            # Test matrix addition
            print("Result of matrix addition:")
            if matrix1.Sparse_add(matrix2):
                print(matrix1.Sparse_add(matrix2).todense())
            print()

            # Test matrix subtraction
            print("Result of matrix subtraction:")
            if matrix1.Sparse_sub(matrix2):
                print(matrix1.Sparse_sub(matrix2).todense())
            print()

            #Test norms
            print('Dense l1 norm')
            print(matrixa.norm('l1'))
            print('Dense l2 norm')
            print(matrixa.norm('l2'))
            print('Dense lif norm')
            print(matrixa.norm('linf'))
            print()

            print('Sparse l1 norm')
            print(matrix1.norm('l1'))
            print('Sparse l2 norm')
            print(matrix1.norm('l2'))
            print('Sparse linf norm')
            print(matrix1.norm('linf'))
            print()

            # Test compute_eigenvalues for dense matrix
            dense_eigenvalues = matrixa.compute_eigenvalues()
            print('Dense eigenvalues')
            print(dense_eigenvalues)
            print()

            # Test compute_eigenvalues for sparse matrix
            sparse_eigenvalues = matrix1.compute_eigenvalues()
            print('Sparse eigenvalues')
            print(sparse_eigenvalues)
            print()

            # Test compute_svd for dense matrix
            dense_U, dense_S, dense_Vt = matrixa.compute_svd()
            print('Dense SVD decomposition')
            print(f'U = {dense_U}')
            print(f"S = {dense_S}")
            print(f"Vt = {dense_Vt}")
            print()
            

            # Test compute_svd for sparse matrix
            sparse_U, sparse_S, sparse_Vt = matrix1.compute_svd()
            print('Dense SVD decomposition')
            print(f'U = {sparse_U}')
            print(f"S = {sparse_S}")
            print(f"Vt = {sparse_Vt}")
            print()  
            
        #Checks if the user runs the part Graph    
        elif sys.argv[2].lower() == 'graph':
            print('Now generating a graph comparing sparse to dense operations')
            matrix_size = int(sys.argv[3])
            #Look if the user specified a sparsity or not
            if len(sys.argv)==5:
                dense_times, sparse_times, operations = Graph.measure_time_operations(matrix_size, float(sys.argv[4]))
            else:
                dense_times, sparse_times, operations = Graph.measure_time_operations(matrix_size)
            Graph.barchart_performance_comparaison(dense_times, sparse_times, operations, matrix_size)

        #Checks if the user runs the part Plot
        elif sys.argv[2].lower() == 'plot':
            print('Now generating a plot comparing the matrix class functions performances depending on the matrix size')
            print(sys.argv[-1])
            matrix_sizes=[]
            #Look if the user specified a sparsity or not
            if float(sys.argv[-1]) < 1:
                for i in sys.argv[3:-1]:
                    matrix_sizes.append(int(i))
                Graph.globalplotcomparison(matrix_sizes, float(sys.argv[-1]))
            else:
                for i in sys.argv[3:]:
                    matrix_sizes.append(int(i))
                Graph.globalplotcomparison(matrix_sizes)

        #Checks if the user runs the part Profile    
        elif sys.argv[2].lower() == 'profile':
            profiler.profiler_all_matrices_functions()

        else:
            print('The third argument of your query does not respect the README guidelines')
    
    #Checks if the user requires a function in part 2
    elif sys.argv[1].lower() == 'part2':
        print('-----PART 2-----')

        #Checks if the user runs the part Transformation
        if sys.argv[2].lower() == 'transformation':
            print('Performing transformation')
            matrix = np.matrix([[1, 0, 3], [0, 5, 8], [4, 1, 9], [4, 1, 8]]) #the user inputs their matrix or array, this is a test case
            #we ran multiple test cases, and it seems to work for both square, non-square and dense matrices
            trans = module21.transformation(matrix) #we call the function from the module to be applied to the inputed matrix
            print ("Exercise1:\n", trans) #print the result
            
        #Checks if the user runs the part Processing
        elif sys.argv[2].lower() == 'processing':
            print('Performing processing')
            matrix = module21.random_generator(int(sys.argv[3]), 0, 10) #again, the user chooses the maximum size of the matrix, and the range of values
            print("Exercise2:", matrix) #print the result

        #Checks if the user runs the part SVDPlot
        elif sys.argv[2].lower() == 'svdplot':
            print('Performing SVDPlot')
            rating_matrix_keep = module21.random_generator(int(sys.argv[3]),0,10) #this line is important, because we need to keep the same randomly generated matrix in 2.3 and 2.4 
            singular_values = module21.SVD(rating_matrix_keep)
            print ("Exercise3:", singular_values) #print the result

        #Checks if the user runs the part SVD  
        elif sys.argv[2].lower() == 'svd':
            print('Performing SVD')
            rating_matrix_keep = module21.random_generator(int(sys.argv[3]),0,10) #this line is important, because we need to keep the same randomly generated matrix in 2.3 and 2.4 
            uv = module21.dimensionality_reduction(rating_matrix_keep)
            print ("Exercise4:", uv) #print the result

        #Checks if the user runs the part Recommend
        elif sys.argv[2].lower() == 'recommend':
            print('Performing recommend')
            recommended_movies = module21.recommend() #the user has nothing to do, since everything is defined within the function.
            print ("Exercise5:", recommended_movies)
        else:
            print('The third argument of your query does not respect the README guidelines')
    #Error message if the user requires something neither in part 1 or 2
    else:
        print("The second argument of your query does not respect the README guidelines")

main()
