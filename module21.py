import numpy as np
import random as rand
import scipy as sp
import Matrixclass as matrixclasses 
import numpy.linalg as linalg
import scipy.linalg as linalg
import time
import matplotlib.pyplot as plt

#2.1
#we want to transform an original full matrix into a binary matrix 
def transformation(matrix): #define a transformation function, with a parameter matrix being inputed by the user in the main
    full_matrix = np.array(matrix) #in case we have to input a dense matrix, this converts the entered matrix into an array, for it to be compatible with the integer treshold:
    treshold = 5 #condition: 1 if x>5, otherwise it is 0
    #we define the lambda transformation function:
    transformation_function = np.vectorize(lambda #np.vectorize  
                                           x: 1 if x > treshold
                                           else 0) #make sure the conditions are clear
    binary_matrix = transformation_function(full_matrix) #we apply the transformation function to the original matrix
    rows, columns = binary_matrix.shape #we need to retrieve the rows and columns to accomodate dense matrices, according to our class defined in part 1
    return matrixclasses.DenseMatrix(rows, columns, binary_matrix) #we return the binary version of the matrix, and it also works for dense matrices

#2.2
#we want to generate random non-square matrixes with more rows than columns
def random_generator(max_size, min_value, max_value): #define the function to randomnly generate matrices, the user needs to input the maximum size they want the matrix to be, and the range of values it can take
    rows =rand.randint(2,max_size) #the number of rows is random, as long as there is more than one up to the maximum size
    columns =rand.randint(1, rows) #the number of columns is also random, as long as there is at least one and up to one less than the row number
    #here the code is more confusing because we have to use the densematrix class from the part 1.
    #so we do: 
    random_data = [[rand.randint(min_value, max_value) for _ in range(columns)] for _ in range(rows)] #we generate random data, in the value range set by the user, and within the rows and columns range as well
    #we have to generate the data before creating the matrix when dealing with dense matrices.
    generated_matrix = matrixclasses.DenseMatrix(rows, columns, random_data) #now we can generate the normal matrix, with all the data already created.
    binary_generated_matrix = transformation(generated_matrix.densematrix) #we call the transformation function on this matrix. 
    #This is why we had to make sure the code 2.1 was adapted to dense matrices.
    print (generated_matrix.densematrix) #we print the original matrix.
    return (binary_generated_matrix.densematrix) #and return the final binary representation.
    
#2.3
#let's perform SVD on the rating matrix from scipy linalg and then numpy linalg
#we first want to generate a rating matrix to perform SVD on
def SVD(rating_matrix): #we define the SVD function with a randomly generated rating_matrix as parameter
    start_time = time.time() #we set the time to compute the differences
    U_scipy, s_scipy, VT_scipy = sp.linalg.svd(rating_matrix, full_matrices = True) #we compute SVD using scipy
    end_time = time.time()
    scipy_time = end_time-start_time #and we compute the total time

    start_time = time.time() #same logic
    U_numpy, s_numpy, VT_numpy = np.linalg.svd(rating_matrix, full_matrices = True) #using numpy
    end_time = time.time()
    numpy_time = end_time-start_time
    #now we compare the two time values
    if numpy_time>scipy_time: #if numpy takes more time:
        print (f"The quickest computation time for SVD is the scipy time, of {scipy_time} seconds.")
        singular_values = s_scipy #and we keep the best performance singular values
        # return singular_values (if we want to show them, we untoggle line comment)
    elif scipy_time>numpy_time: #other case: scipy takes more time
        print (f"The quickest computation time for SVD is the numpy time, of {numpy_time} seconds.")
        singular_values = s_numpy
        # return singular_values
    #now we plot the graph:
    plt.bar(range(1, len(singular_values)+1), singular_values) #we do a bar plot to see the contrast more easily, from 1 to the amount of singular values there are, incremented for each singular value
    plt.title('Scree Plot') #plots of singular values are called 'scree plots'
    plt.xlabel('Principal Component') #define the x and y axes
    plt.ylabel('Singular Value')
    plt.show() #display the graph

#2.4
def dimensionality_reduction(rating_matrix): #we define this function, based on the same rating matrix as the 2.3 question
    #this function is simpler because we use the same formulas as the previous question, only we focus on different elements of SVD
    #we use sp.linalg.svd based on multiple test runs of the question 2.3: most of the time, scipy tends to be faster than numpy.
    U_scipy, s_scipy, VT_scipy = sp.linalg.svd(rating_matrix, full_matrices = True) #the SVD formula is the same.
    selected_values = np.where(s_scipy>2) #we select a relevant criteria. To do this, we are also based on test runs from 2.3: most singular values are superior to 2, or 3, so we eliminate the smallest, most irrelevant values.
    # singular_values become the selected_values
    # we talk about columns for U so we will study the VT rows
    U_values = U_scipy[:,selected_values] #we list the U values along the columns
    VT_values = VT_scipy[selected_values,:] #and VT values along the rows
    return U_values, VT_values #we return the matrices

#2.5
def recommend(): #we define a function based on the pseudocode structure
    #we start by computing the needed parameters. We choose not to put them into the function as parameters for simplicity, and so that everything is included in one function, and not in the main.
    rating_matrix = random_generator(50,0,10) #we generate a random matrix to simulate a user's movie experience and taste
    U_scipy, s_scipy, VT_scipy = sp.linalg.svd(rating_matrix, full_matrices=True) #we do SVD to get the VT matrix
    VT = np.matrix(VT_scipy)
    liked_movie_index = rand.randint(0, 2) #we randomly chose a liked movie index
    Selected_movies_num = rand.randint(0, 51) #and a selected movies number. We choose low numbers to accelerate the results.
    #now we can generate the recommendations list
    recommended = [] #initialize the list
    for i in range(len(VT)): #within the size of the matrix
        if i != liked_movie_index: #for every movie except the liked one we are basing the algorithm on:
            similarity = np.dot(VT[i, :], VT[liked_movie_index, :].T)[0, 0] #we compute the similarity do the dot product bewteen the movies index and the liked movie index
            #we need to transpose it so that the dimensions match, and so that we can actually compute the dot product
            recommended.append([i, similarity]) #we add to the list an element composed of the index of the movie and its similarity rate
    final_recommendations = sorted(recommended) #we sort the recommendations in ascending order (index from 0 to however many recommended movies we have).
    #sometimes the output is an empty list because the conditions aren't met, so we check:
    if final_recommendations:
        return final_recommendations[:Selected_movies_num] #if there are recommendations in the list, then it returns the normal list.
    else:
        return None #if there aren't, it just returns None as in No recommendations