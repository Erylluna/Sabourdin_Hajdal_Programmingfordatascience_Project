import numpy as np
from scipy.sparse import csc_array, csc_matrix
from scipy.linalg import svd 
from scipy.sparse.linalg import eigs, svds, spsolve

class Matrix:
    def __init__(self, nbrows, nbcolumns):
        '''Function that initiates the matrix class and sets its characteristic values'''
        self.nbrows = nbrows
        self.nbcolumns = nbcolumns
        #creates an empty 0s matrix 
        ### this function is useful within other functions definitions but not necessarily outside
        self.initialmatrix = [[0] * nbcolumns for _ in range(nbrows)]

    def get_size(self):
        '''Function that returns the size of any matrix'''
        return (self.nbrows, self.nbcolumns)
    
    def norm(self, lx = 'l1'):
        '''Function that computes the norm of a matrix, wether it being sparse or dense
        input: lx = the type of norm deriser (l1 by default, but can be l2 or linf)
        output: lx_norm (float)
        '''
        #checks wether the method is used on a Dense matrix
        if isinstance(self, DenseMatrix):
            #checks the type of norm asked
            if lx.lower() == 'l1':
                ###computes the norm l1 for dense matrix###
                #initiates a variable to hold the largest column sum
                total_sum = 0
                #compute the column sum for each column
                for i in range(self.nbcolumns):
                    column_sum = 0
                    for j in range(self.nbrows):
                        column_sum += abs(self.densematrix[j][i])
                    if column_sum>=total_sum:
                        #select the column sum with the maximum value
                        total_sum = column_sum
                return total_sum
            elif lx.lower() == 'l2':
                ###computes the norm l2 for dense matrix###
                #initiates a variable to hold the euclidian norm
                total_sum = 0
                #compute the sum of the square of all values in the matrix
                for i in range (self.nbrows):
                    for j in range(self.nbcolumns):
                        total_sum+=(self.densematrix[i][j])**2
                #compute the euclidian norm: the square root of the sum of the square of all values in the matrix
                total_sum = total_sum**0.5
                return total_sum
            elif lx.lower() == 'linf':
                '''Function that computes the norm of a matrix, wether it being sparse or dense
                input: lx = the type of norm deriser (l1 by default, but can be l2 or linf)
                output: lx_norm (float)
                '''
                ###computes the norm linf for dense matrix###
                #initiates a variable to hold the largest row sum
                total_sum = 0
                #compute the row sum for each column
                for i in range(self.nbrows):
                    row_sum = 0
                    for j in range(self.nbcolumns):
                        row_sum += abs(self.densematrix[i][j])
                    if row_sum>=total_sum:
                        #select the row sum with the maximum value
                        total_sum = row_sum
                return total_sum
            else:
                #Error message if the norm is not in the defined types
                print('The type of norm can only be either l1, l2 or linf')

               
        elif isinstance(self, SparseMatrix):
            if lx.lower() == 'l1':
                ###computes the norm l1 for sparse matrix###
                #initiates a variable to hold the largest column sum
                total_sum = 0
                for i in range(self.nbcolumns):
                    #compute the column sum for each column
                    column_sum = 0
                    for j in self.non_empty_columns:
                        if j == i:
                            column_sum += abs(self.values[i])                       
                    #select the column sum with the maximum value
                    if column_sum>=total_sum:
                        total_sum = column_sum
                return total_sum
            elif lx.lower() == 'l2':
                ###computes the norm l2 for dense matrix###
                #initiates a variable to hold the euclidian norm
                total_sum = 0
                #compute the sum of the square of all values in the matrix
                for i in self.values:
                    total_sum+= i ** 2
                #compute the euclidian norm: the square root of the sum of the square of all values in the matrix
                total_sum = total_sum ** 0.5
                return total_sum
            elif lx.lower() == 'linf':
                ###computes the norm linf for dense matrix###
                #initiates a variable to hold the largest row sum
                total_sum = 0
                #compute the row sum for each column
                for i in range(self.nbrows):
                    row_sum = 0
                    for j in self.non_empty_rows:
                        if j == i:
                            row_sum += abs(self.values[i])
                    if row_sum>=total_sum:
                        #select the row sum with the maximum value
                        total_sum = row_sum
                return total_sum
            else:
                #Error message if the norm is not in the defined types
                print('The type of norm can only be either l1, l2 or linf')
                return None
        else:
            #Error message if the norm is calculated on neither a Dense Matrix nor a Sparse Matrix
            print ('To compute its norm, the object must be either a sparse matrix or a dense matrix.')
        return None
    
    def compute_eigenvalues(self, vectors = False):
        '''Function that computes the eigenvalues of a matrix, wether it being sparse or dense
        input: vectors = bool (define wether you want to retrieve the eigenvectors along with the eigenvalues)
        output: eigenvalues (array), optional: eigenvectors (array)
        '''
        #Check if the matrix is a square matrix
        if self.nbrows != self.nbcolumns:
            print("Eigenvalues can only be computed for square matrices.")
            return None
        #Check if the object is a Dense matrix
        if isinstance(self, DenseMatrix):
            #Compute the eigenvalues & vectors using numpy
            eigenvalues, eigenvectors = np.linalg.eig(np.array(self.densematrix))
        #Check is the object is a Sparse matrix
        elif isinstance(self, SparseMatrix):
            #Compute the eigenvalues & vectors using scipy
            eigenvalues, eigenvectors = eigs(csc_array((self.values, (self.non_empty_rows, self.non_empty_columns)), (self.nbrows, self.nbcolumns)))
        else: 
            #Error message if the eigenvalues are calculated on neither a Dense Matrix nor a Sparse Matrix
            print('The object needs to be a Sparse or a Dense matrix to compute its eigenvalues')
        if vectors == 'True':
            #return the vectors if asked
            return eigenvalues, eigenvectors
        return eigenvalues

    def compute_svd(self):
        '''Function that does the svd decomposition of a matrix, wether it being sparse or dense
        input: none
        output: U (array), S(array), Vt (array)
        '''
        #Check if the object is a Dense matrix
        if isinstance(self, DenseMatrix):
            #Compute the svd decomposition using numpy
            U, S, Vt = svd(np.array(self.densematrix))
        #Check is the object is a Sparse matrix
        elif isinstance(self, SparseMatrix):
            #Compute the eigenvalues & vectors using scipy
            U, S, Vt = svds(csc_array((self.values, (self.non_empty_rows, self.non_empty_columns)), (self.nbrows, self.nbcolumns)))
        else: 
            #Error message if the svd decomposition is done on neither a Dense Matrix nor a Sparse Matrix
            print('The object needs to be a Sparse or a Dense matrix to compute its eigenvalues')
        
        return U, S, Vt
    
    def solvelinearsystem(self, vector_leftside):
        '''Function that solves a linear system of the form: Matrix * x = vector_leftside, wether the matrix being sparse or dense
        input: vectors_leftside = the vector on the left side of the equation
        output: solutions(array) = the values of x that satisfies Matrix * x = vector_leftside
        '''
        #Check if the object is a Dense matrix
        if isinstance(self, DenseMatrix):
            #resolve the linear system using numpy
            solutions= np.linalg.solve(self.densematrix, vector_leftside)
            return solutions
        #Check is the object is a Sparse matrix
        elif isinstance(self, SparseMatrix):
            #esolve the linear system using scipy
            solutions= spsolve(csc_matrix((self.values, (self.non_empty_rows, self.non_empty_columns)), (self.nbrows, self.nbcolumns)), vector_leftside)
            return solutions
        else: 
            #Error message if the system is done on neither a Dense Matrix nor a Sparse Matrix
            print('The object needs to be a Sparse or a Dense matrix to resolve the linear system')


    
    
class DenseMatrix(Matrix):
    def __init__(self, nbrows, nbcolumns, matrix):
        '''Function that initiates the DenseMatrix class and sets its characteristic values'''
        Matrix.__init__(self, nbrows, nbcolumns)
        self.densematrix = np.array(matrix)

    def tosparse(self):
        '''Function that takes a dense matrix and returns all the data needed to turn it into a Sparse matrix 
        input: none
        output: [non_empty_rows, non_empty_columns, values] -> an array such that the sparse matrix can be defined as: SparseMatrix(self.nbrows, self.nbcolumns, array[0], array[1], array[2])'''
        #Sets empty arrays to hold the data usefull in making a sparse matrix
        non_empty_rows = [] 
        non_empty_columns = []
        values = []
        #Extract all the non-zero values from the dense matrix, while storing its indexes
        for i in range(self.nbrows):
            for j in range(self.nbcolumns):
                if self.densematrix[i][j] != 0:
                    non_empty_rows.append(i)
                    non_empty_columns.append(j)
                    values.append(self.densematrix[i][j])
                  
        return [non_empty_rows, non_empty_columns, values]


    def Dense_add(self, other):
        '''Function that adds a dense matrix to another
        input: other (densematrix)
        output: Densematrix -> the dense matrix resulting of the sum of the two matrices'''
        #Check that the arguments are Dense matrices
        if not isinstance(other, DenseMatrix):
            print("To be added, the two objects must be dense matrices")
            return None
        #Checks that both matrices have the same size
        if self.get_size() != other.get_size():
            print("The two dense matrices must have the same size to be added")
            return None
        
        #Creates an matrix of the size of the inputed one, full of 0s
        Summatrix = [[0] * self.nbcolumns for _ in range(self.nbrows)]
        #Fill the matrix with the sum of each component of the dense matrices, component to component
        for i in range(self.nbrows):
            for j in range(self.nbcolumns):
                Summatrix[i][j] = self.densematrix[i][j] + other.densematrix[i][j]
        
        return DenseMatrix(self.nbrows, self.nbcolumns, Summatrix)

    def Dense_subtract(self, other):
        '''Function that subs a dense matrix to another
        input: other (densematrix) -> the one which will be subtracted to the first one
        output: Densematrix -> the dense matrix resulting of the subtraction of the two matrices'''
        #Check that the arguments are Dense matrices
        if not isinstance(other, DenseMatrix):
            print("To be subtracted from one another, the two objects must be dense matrices")
            return None
        #Checks that both matrices have the same size
        if self.get_size() != other.get_size():
            print("The two dense matrices must have the same size for one to be subtracted from the other")
            return None
        
        #Creates an matrix of the size of the inputed one, full of 0s
        Submatrix = [[0] * self.nbcolumns for _ in range(self.nbrows)]
        #Fill the matrix with the sum of each component of the dense matrices, component to component
        for i in range(self.nbrows):
            for j in range(self.nbcolumns):
                Submatrix[i][j] = self.densematrix[i][j] - other.densematrix[i][j]
        
        return DenseMatrix(self.nbrows, self.nbcolumns, Submatrix)

    def Dense_multiply(self, other):
        '''Function that multiply a dense matrix with another
        input: other (densematrix) -> can be of size (1,2) : allows for vector multiplicatio
        output: Densematrix -> the dense matrix resulting of the multiplication of the two matrices'''
        #Check that the arguments are Dense matrices
        if not isinstance(other, DenseMatrix):
            print("To be multiplied to one another, the two objects must be dense matrices")
            return None
        #Check that the matrices' dimensions allow multiplication
        if self.nbcolumns != other.nbrows:
            print("The number of columns in the first matrix must match the number of rows in the second matrix for multiplication.")
            return None
        
        #Creates an matrix of size (columnsother, rowsself), full of 0s
        result = [[0] * other.nbcolumns for _ in range(self.nbrows)]
        #Fill the matrix with the sum of the product of each component of the rows of self with the column of other
        for i in range(self.nbrows):
            for j in range(other.nbcolumns):
                for k in range(self.nbcolumns):
                    result[i][j] += self.densematrix[i][k] * other.densematrix[k][j]
        
        return DenseMatrix(self.nbrows, other.nbcolumns, result)
    #Works too with vectors as they can be written like dense matrices of dimension (1, m)
    
    def __str__(self):
        s = ""
        for r in self.densematrix:
            for v in r:
                s += f"{v}, "
            s += "\n"
        return s
       
class SparseMatrix(Matrix):
    # For defining sparse matrices, we use the Compressed Sparse Column (CSC) format
    def __init__(self, nbrows, nbcolumns, non_empty_rows, non_empty_columns, values):
        '''Function that initiates the SparseMatrix class and sets its characteristic values'''
        Matrix.__init__(self, nbrows, nbcolumns)
        self.non_empty_rows = non_empty_rows
        self.non_empty_columns = non_empty_columns
        self.values = values
        #creates a list following the CSC format for the spare matrices
        self.CSCmatrix = [non_empty_rows, non_empty_columns, values]
    
    def todense(self):
        '''Function that turns a Sparse matrix to a Dense matrix
        input: none
        output: Densematrix (np.array -> so that a dense matrix can be initiated with: DenseMatrix(self.nbrows, self.nbcolumns, Densematrix))'''
        #creating a variable that will follow the indexes of the values within the value list
        k=0
        #creating a matrix of the size of the sparse matrix full of 0s
        densematrix = self.initialmatrix
        #setting all the values of the sparse matrix to their indexes in the dense matrix
        for i, j in zip(self.non_empty_rows, self.non_empty_columns):
            densematrix[i][j] = self.values[k]
            k+=1
        return np.array(densematrix)

    def getsparsity(self):
        '''Function that returns the sparsity of the matrix
        input: none
        output: sparsity (int)'''
        sparsity = len(self.values)/(self.nbrows * self.nbcolumns)
        return sparsity
    
    def Sparse_add(self, other):
        '''Function that adds a sparse matrix to another
        input: other (sparsematrix)
        output: Sparsematrix -> the sparse matrix resulting of the sum of the two matrices'''
        #Check that the other object is also a sparse matrix
        if not isinstance(other, SparseMatrix):
            print("To be added, the two objects must be sparse matrices")
            return None
        #Check that the sizes are the same (but without passing through get size as it slows the process)
        if self.nbrows != other.nbrows or self.nbcolumns != other.nbcolumns:
            print("The two dense matrices must have the same size to be added")
            return None
        
        #Create arrays to hold the indexes of the summed Sparse matrix values and its values
        result_non_empty_rows = []
        result_non_empty_columns = []
        result_values = []

        #Look for every index in the other sparse matrix that matches the index of the self sparse matrix
        for i,j,m in zip(self.non_empty_rows, self.non_empty_columns, self.values):
            ver = 0
            for k, l, n in zip(other.non_empty_rows, other.non_empty_columns, other.values):
                if (i,j)==(k,l):
                    #Append the index to the new indexes lists, and the sum of the values at this index in the matrices to the new values lists
                    result_non_empty_rows.append(i)
                    result_non_empty_columns.append(j)
                    result_values.append(m+n)
                    ver = 1
                    break
            #For all the indexes in the self sparse matrix that has no match in the other, append its indexes and its values to the new lists 
            if ver == 0:
                result_non_empty_rows.append(i)
                result_non_empty_columns.append(j)
                result_values.append(m)
            
        #For all the indexes in the other sparse matrix that has no match in the self, append its indexes and its values to the new lists 
        for i,j,n in zip(other.non_empty_rows, other.non_empty_columns, other.values):
            ver = 0
            #Check that it has no matches
            for k, l in zip(result_non_empty_rows, result_non_empty_columns):
                if (i,j)==(k,l):
                    ver = 1
                    break
            if ver == 0:
                result_non_empty_rows.append(i)
                result_non_empty_columns.append(j)
                result_values.append(n)
        return SparseMatrix(max(result_non_empty_rows)+1, max(result_non_empty_columns)+1, result_non_empty_rows, result_non_empty_columns, result_values)
    
    def Sparse_sub(self, other):
        '''Function that subtract a sparse matrix to another
        input: other (sparsematrix)
        output: Sparsematrix -> the sparse matrix resulting of the subtraction of the two matrices'''
        #Check that the other object is also a sparse matrix
        if not isinstance(other, SparseMatrix):
            print("To be subtracted from one another, the two objects must be sparse matrices")
            return None
        #Check that the sizes are the same (but without passing through get size as it slows the process)
        if self.nbrows != other.nbrows or self.nbcolumns != other.nbcolumns:
            print("The two dense matrices must have the same size to be subtracted from one another")
            return None
        
        #Create arrays to hold the indexes of the subtracted Sparse matrix values and its values
        result_non_empty_rows = []
        result_non_empty_columns = []
        result_values = []

        #Look for every index in the other sparse matrix that matches the index of the self sparse matrix
        for i,j,m in zip(self.non_empty_rows, self.non_empty_columns, self.values):
            ver = 0
            for k, l, n in zip(other.non_empty_rows, other.non_empty_columns, other.values):
                if (i,j)==(k,l):
                    #Append the index to the new indexes lists, and the difference of the values at this index in the matrices to the new values lists
                    result_non_empty_rows.append(i)
                    result_non_empty_columns.append(j)
                    result_values.append(m-n)
                    ver = 1
                    break
            #For all the indexes in the self sparse matrix that has no match in the other, append its indexes and its values to the new lists 
            if ver == 0:
                result_non_empty_rows.append(i)
                result_non_empty_columns.append(j)
                result_values.append(m)

        #For all the indexes in the other sparse matrix that has no match in the self, append its indexes and minus its values to the new lists 
        for i,j,n in zip(other.non_empty_rows, other.non_empty_columns, other.values):
            ver = 0
            #Check that it has no matches
            for k, l in zip(result_non_empty_rows, result_non_empty_columns):
                if (i,j)==(k,l):
                    ver = 1
                    break
            if ver == 0:
                result_non_empty_rows.append(i)
                result_non_empty_columns.append(j)
                result_values.append(-n)
        return SparseMatrix(max(result_non_empty_rows)+1, max(result_non_empty_columns)+1, result_non_empty_rows, result_non_empty_columns, result_values)

    def Sparse_multiply(self, other):
        '''Function that multiply a sparse matrix to another
        input: other (sparsematrix)
        output: Sparsematrix -> the sparse matrix resulting of the product of the two matrices'''
        #Check that the other object is also a sparse matrix
        if not isinstance(other, SparseMatrix):
            print("To be multiplied, the two objects must be sparse matrices")
            return None
        #Check that the dimensions of the matrices allow for multiplication
        if self.nbcolumns != other.nbrows:
            print("The number of columns in the first matrix must match the number of rows in the second matrix for multiplication.")
            return None

        #Create arrays to hold the indexes of the multiplied Sparse matrix values and its values
        result_values = []
        result_non_empty_rows = []
        result_non_empty_columns = []

        # Check for common column in first matrix and row in second matrix, append their index and the product of them to one another
        for row1, col1, val1 in zip(self.non_empty_rows, self.non_empty_columns, self.values):
            for row2, col2, val2 in zip(other.non_empty_rows, other.non_empty_columns, other.values):
                if col1 == row2:  
                    result_non_empty_rows.append(row1)
                    result_non_empty_columns.append(col2)
                    result_values.append(val1 * val2)

        return SparseMatrix(self.nbrows, other.nbcolumns, result_non_empty_rows, result_non_empty_columns, result_values)