import Matrixclass as M
import numpy as np
import time
import matplotlib.pyplot as plt

# Define your Matrix class and its methods here

# Function to generate random non-singular matrices of size n x n
def create_nonsingular__dense_squarematrix(size):
    '''Function that creates a non-singular dense square matrix
    
    input: the size of the matrix
    
    output: Matrix (type: dense matrix)'''
    #set a variable k that checks wether the matrix is non singular
    k=0
    while k == 0:
        #while the variable is at its initial value, create a random square matrix 
        matrix = np.random.rand(size, size)
        #check if it is nonsingular
        if np.linalg.matrix_rank(matrix) == size:
            #if it is, change the variable k
            k=1
    return M.DenseMatrix(size, size, matrix)

def create__sparse_squarematrix(matrix_size, sparsity=0.3):
    '''Function that creates a sparse square matrix
    
    input: matrix_size = the size of the matrix 
    sparsity = desired sparsity (0.3 by default)

    output: Matrix (type: sparse matrix)
    '''
    #create a random square matrix of the desired sparsity
    num_elements = int(matrix_size * matrix_size * sparsity)
    non_empty_rows = np.random.randint(0, matrix_size, num_elements)
    non_empty_columns = np.random.randint(0, matrix_size, num_elements)
    values = np.random.rand(num_elements)
    return M.SparseMatrix(matrix_size, matrix_size, non_empty_rows, non_empty_columns, values)

def create_nonsingular__sparse_squarematrix(matrix_size, sparsity=0.3):
    '''Function that creates a non_singular sparse square matrix
    
    input: matrix_size = the size of the matrix 
    sparsity = desired sparsity (0.3 by default)

    output: Matrix (type: sparse matrix)
    '''
    #set a variable k that checks wether the matrix is non singular
    k=0
    while k == 0:
        sparse_matrix = create__sparse_squarematrix(matrix_size, sparsity)
        if np.linalg.matrix_rank(sparse_matrix.todense()) == matrix_size:
            k=1
    return sparse_matrix


def measure_time_operations(matrix_size, sparsity = 0.3, profiler=False):
    '''Function that measures the time taken by each operation in the Matrix Class
    
    input: matrix_size = the size of the matrix 
    sparsity = desired sparsity (0.3 by default)

    output: dense_time: array of the times taken to do the operations for the Dense Matrix child-class
    sparse_time: array of the times taken to do the operations for the Sparse Matrix child-class
    operations: an array containing the list of the names of the operations in the Matrix class
    '''
    #generate two random non singular dense square matrices
    dense_matrix1 = create_nonsingular__dense_squarematrix(matrix_size)
    dense_matrix2 = create_nonsingular__dense_squarematrix(matrix_size)

    # Measure the time taken by the operation Dense_Add
    start_time = time.time()
    dense_matrix1.Dense_add(dense_matrix2)
    end_time = time.time()
    Dense_Add_execution_time = end_time - start_time
    print("Dense_Add Execution Time:", Dense_Add_execution_time, "seconds")

    # Measure the time taken by the operation Dense_Subtract
    start_time = time.time()
    dense_matrix1.Dense_subtract(dense_matrix2)
    end_time = time.time()
    Dense_Subtract_execution_time = end_time - start_time
    print("Dense_Subtract Execution Time:", Dense_Subtract_execution_time, "seconds")

    # Measure the time taken by the operation Dense_Multiply
    start_time = time.time()
    dense_matrix1.Dense_multiply(dense_matrix2)
    end_time = time.time()
    Dense_Multiply_execution_time = end_time - start_time
    print("Dense_Multiply Execution Time:", Dense_Multiply_execution_time, "seconds")

    # Measure the time taken by the operation norm l1 for DenseMatrix
    start_time = time.time()
    dense_matrix1.norm('l1')
    end_time = time.time()
    dense_norm_l1_execution_time = end_time - start_time
    print("Dense Norm l1 Execution Time:", dense_norm_l1_execution_time, "seconds")

    # Measure the time taken by the operation norm l2 for DenseMatrix
    start_time = time.time()
    dense_matrix1.norm('l2')
    end_time = time.time()
    dense_norm_l2_execution_time = end_time - start_time
    print("Dense Norm l2 Execution Time:", dense_norm_l2_execution_time, "seconds")

    # Measure the time taken by the operation norm linf for DenseMatrix
    start_time = time.time()
    dense_matrix1.norm('linf')
    end_time = time.time()
    dense_norm_linf_execution_time = end_time - start_time
    print("Dense Norm linf Execution Time:", dense_norm_linf_execution_time, "seconds")

    # Measure the time taken by the operation compute_eigenvalues for DenseMatrix
    start_time = time.time()
    dense_matrix1.compute_eigenvalues()
    end_time = time.time()
    dense_eigenvalues_execution_time = end_time - start_time
    print("Dense Eigenvalues Execution Time:", dense_eigenvalues_execution_time, "seconds")
    
    # Measure the time taken by the operation compute_svd for DenseMatrix
    start_time = time.time()
    dense_matrix1.compute_svd()
    end_time = time.time()
    dense_svd_execution_time = end_time - start_time
    print("Dense SVD Execution Time:", dense_svd_execution_time, "seconds")

    # Measure the time taken by the operation solvelinearsystem for DenseMatrix
    start_time = time.time()
    dense_matrix1.solvelinearsystem(np.random.rand(matrix_size, 1))
    end_time = time.time()
    dense_solve_linear_system_execution_time = end_time - start_time
    print("Dense Solve Linear System Execution Time:", dense_solve_linear_system_execution_time, "seconds")

    #generate two random non singular sparse square matrices
    sparse_matrix1 = create_nonsingular__sparse_squarematrix(matrix_size, sparsity)
    sparse_matrix2 = create_nonsingular__sparse_squarematrix(matrix_size, sparsity)

    # Measure the time taken by the operation Sparse_Add
    start_time = time.time()
    sparse_matrix1.Sparse_add(sparse_matrix2)
    end_time = time.time()
    sparse_add_execution_time = end_time - start_time
    print("Sparse_Add Execution Time:", sparse_add_execution_time, "seconds")

    # Measure the time taken by the operation Sparse_Subtract
    start_time = time.time()
    sparse_matrix1.Sparse_sub(sparse_matrix2)
    end_time = time.time()
    sparse_subtract_execution_time = end_time - start_time
    print("Sparse_Subtract Execution Time:", sparse_subtract_execution_time, "seconds")

    # Measure the time taken by the operation Sparse_Multiply
    start_time = time.time()
    sparse_matrix1.Sparse_multiply(sparse_matrix2)
    end_time = time.time()
    sparse_multiply_execution_time = end_time - start_time
    print("Sparse_Multiply Execution Time:", sparse_multiply_execution_time, "seconds")

    # Measure the time taken by the operation norm l2 for SparseMatrix
    start_time = time.time()
    sparse_matrix1.norm(lx='l1')
    end_time = time.time()
    sparse_norm_l1_execution_time = end_time - start_time
    print("Sparse Norm l1 Execution Time:", sparse_norm_l1_execution_time, "seconds")

    # Measure the time taken by the operation norm l1 for SparseMatrix
    start_time = time.time()
    sparse_matrix1.norm(lx='l2')
    end_time = time.time()
    sparse_norm_l2_execution_time = end_time - start_time
    print("Sparse Norm l2 Execution Time:", sparse_norm_l2_execution_time, "seconds")

    # Measure the time taken by the operation norm linf for SparseMatrix
    start_time = time.time()
    sparse_matrix1.norm(lx='linf')
    end_time = time.time()
    sparse_norm_linf_execution_time = end_time - start_time
    print("Sparse Norm linf  Execution Time:", sparse_norm_linf_execution_time, "seconds")

    # Measure the time taken by the operation compute_eigenvalues for SparseMatrix
    start_time = time.time()
    sparse_matrix1.compute_eigenvalues()
    end_time = time.time()
    sparse_eigenvalues_execution_time = end_time - start_time
    print("Sparse Eigenvalues Execution Time:", sparse_eigenvalues_execution_time, "seconds")
       
    # Measure the time taken by the operation compute_svd for SparseMatrix
    start_time = time.time()
    sparse_matrix1.compute_svd()
    end_time = time.time()
    sparse_svd_execution_time = end_time - start_time
    print("Sparse SVD Execution Time:", sparse_svd_execution_time, "seconds")

    # Measure the time taken by the operation solvelinearsystem for SparseMatrix
    start_time = time.time()
    sparse_matrix1.solvelinearsystem(np.random.rand(matrix_size, 1))
    end_time = time.time()
    sparse_solve_linear_system_execution_time = end_time - start_time
    print("Sparse Solve Linear System Execution Time:", sparse_solve_linear_system_execution_time, "seconds")

    dense_times = [
        Dense_Add_execution_time, Dense_Subtract_execution_time, Dense_Multiply_execution_time,
        dense_norm_l1_execution_time, dense_norm_l2_execution_time, dense_norm_linf_execution_time,
        dense_eigenvalues_execution_time, dense_svd_execution_time, dense_solve_linear_system_execution_time
    ]
    sparse_times = [
        sparse_add_execution_time, sparse_subtract_execution_time, sparse_multiply_execution_time,
        sparse_norm_l1_execution_time, sparse_norm_l2_execution_time, sparse_norm_l2_execution_time,
        sparse_eigenvalues_execution_time, sparse_svd_execution_time, sparse_solve_linear_system_execution_time
    ]
    operations = [
        'Add', 'Subtract', 'Multiply',
        'Norm l1', 'Norm l2', 'Norm linf',
        'Eigenvalues', 'SVD', 'Solve Linear System'
    ]


    return dense_times, sparse_times, operations

def barchart_performance_comparaison(dense_times, sparse_times, operations, matrix_size=100):
    '''Function that creates a barchart comparing the time taken by a sparse and by a dense matrix of fixed size on similar operations
    
    input: matrix_sizes = the size of the matrices (as an int)
    dense_time: array of the times taken to do the operations for the Dense Matrix child-class
    sparse_time: array of the times taken to do the operations for the Sparse Matrix child-class
    operations: an array containing the list of the names of the operations in the Matrix class
    
    output: None
    '''
    #create a variable to store the labels
    labe = np.arange(len(operations)) 
    #set the width of the bars
    width = 0.35  

    #create the columns for the dense matrix and the one for the sparse matrix on the same graph
    fig, ax = plt.subplots()
    ax.bar(labe - width/2, dense_times, width, label='Dense Matrix')
    ax.bar(labe + width/2, sparse_times, width, label='Sparse Matrix')

    # Add labels, title, x-axis tick labels and legends
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title(f'Execution Time Comparison for Different Operations, for matrix size = {matrix_size}')
    ax.set_xticks(labe)
    ax.set_xticklabels(operations, rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()

    plt.show()
    return None

def globalplotcomparison(matrix_sizes=[20, 40, 60, 80, 100], sparsity=0.3):
    '''Function that creates a plot comparing the time taken by each operations in the matrix class, for Sparse and Dense matrices

    input: matrix_sizes = the sizes of the matrices (as an array of integers)
    sparsity = an optional float indicating the sparsity of the matrix
    
    output: None
    '''
    #manually write the list of the operations within the Matrix 
    # Operations to measure
    operations = [
         'Add', 'Subtract', 'Multiply',
        'Norm l1', 'Norm l2', 'Norm linf',
        'Eigenvalues', 'SVD', 'Solve Linear System'
    ]

    # Lists to store execution times for dense and sparse matrices
    dense_times_list = []
    sparse_times_list = []

    # Measure execution times for each matrix size
    for matrix_size in matrix_sizes:
        # Measure times for dense and sparse matrices
        dense_times, sparse_times, _ = measure_time_operations(matrix_size, sparsity=0.3)
        
        # Append times to respective lists
        dense_times_list.append(dense_times)
        sparse_times_list.append(sparse_times)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot execution times for each operation
    for i in range(len(operations)):
        # Extract dense and sparse times for current operation
        dense_times = [times[i] if len(times) > i else None for times in dense_times_list]
        sparse_times = [times[i] if len(times) > i else None for times in sparse_times_list]

        # Plot dense matrix times
        ax.plot(matrix_sizes, dense_times, label='Dense Matrix: ' + operations[i])
        
        # Plot sparse matrix times with dashed linestyle
        ax.plot(matrix_sizes, sparse_times, label='Sparse Matrix: ' + operations[i], linestyle='--')

    # Add labels, title, and legend
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Global Comparison of Execution Time for Different Matrix Sizes and Operations')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Set log scale for both axes
    
    # Show with a grid
    plt.grid(True)

    # Set matrix sizes as x-axis ticks
    plt.xticks(matrix_sizes, rotation=45)
    
    # Adjust layout
    plt.tight_layout()

    # Display plot
    plt.show()

