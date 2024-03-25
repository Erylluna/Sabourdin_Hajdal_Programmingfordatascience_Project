# Sabourdin_Hajdal_Programmingfordatascience_Project
This code provides useful tools for manipulating matrices and an exemple of how we can use them in a real corporate project

If you want to run the part one, which will consist in a demonstration of the matrix functions for matrices of the size of your choice: 

You can choose which part you want to run:

Part1: Operations on matrices
Operations: Runs matrix operations on one specific matrix (1.1; 1.2; 1.3)
Profile: Open a snakeviz window in your internet browser to show how to optimize the different matrix function (1.4)
Graph: A graph about the performance of the matrix class functions one one specific matrix (1.5)
Plot: A plot comparing the performance of matrix class functions for multiple matrices depending on their sizes (1.5)

Part2: Corporate exemple
Transformation: Returns a binary matrix, which was converted from a non-binary numpy array (2.1)
Processing: Returns a randomly-generated matrix of inputed maximum_size (the matrix will be of this size or smaller) and its conversion to a binary matrix (2.2)
SVDPlot: Informs the user on wether scipy or numpy does the fastest SVD decomposition and plots the singular values on a randomly generated matrix of inputed maximum size (2.3)
SVD: returns the U and Vt matrices of the SVD decomposition of a randomly generated matrix of inputed maximum size (2.4)
Recommend: Returns an array of the most recommended films based on how similar they are to the most liked movies

--------------------------------------------------------------------------------------------------------------------------------
What to run ? 

>>> python3 main.py part1 Operations sq_matrix_size
exemple: python3 main.py part1 Operations 10

>>> python3 main.py part1 Graph sq_matrix_size sparsity
exemple: python3 main.py part1 Graph 10

>>> python3 main.py part1 Plot matrix_sizes sparsity
exemple: python3 main.py part1 Plot 20 40 60 80 100 0.6

>>> python3 main.py part1 Profile
exemple: python3 main.py part1 Profile
(-> no manually changed variables)

>>> python3 main.py part2 Transformation
exemple: python3 main.py part2 Transformation
(-> no manually changed variables)

>>> python3 main.py part2 Processing max_size
exemple: python3 main.py part2 Processing 100

>>> python3 main.py part2 SVDPlot max_size
exemple: python3 main.py part2 SVDPlot 1000

>>> python3 main.py part2 SVD max_size
exemple: python3 main.py part2 SVD 1000

>>> python3 main.py part2 Recommend
exemple: python3 main.py part2 Recommend 
(-> no manually changed variables)

\-> with :
sq_matrix_size being an int of your choice (the graphs calls all the functions in the matrix class and some of them can only work with square matrices)
sparsity being an optional argument (a float of absolute value inferior to one)
matrix_sizes being ints of your choice (you can put multiple integers as long as they are separeted by a space and no other character)
max_size being an int of your choice

