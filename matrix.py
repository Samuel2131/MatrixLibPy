
import numpy as np
import algebra_library as al

class matrix:
    def __preControl(self):
        if(not isinstance(self.__matrix, type(np.array([])))): self.__matrix = np.array(self.__matrix)
        
        try:
            self.__matrix.shape[1]
        except IndexError:
            self.__matrix = self.__matrix.reshape((1,-1))
            
        return self.__matrix
   
    def __init__(self, matrix, sarrus=True, show_GaussMatrix=False, historical=False):
        self.__matrix = matrix
        self.__matrix = self.__preControl()
        
        self.__degree = al.calculateDegree(self.__matrix)
        self.__determinant = al.calculateDeterminant(self.__matrix, sarrus, historical=historical)
        self.__rank = al.calculateRank(self.__matrix, show_gaussMatrix=show_GaussMatrix)
        self.__reverseMatrix = al.calculateReverse(self.__matrix)

    def getMatrix(self):
        return self.__matrix

    def printMatrix(self):
        print(self.__matrix)

    def degree(self):
        return self.__degree

    def rank(self):
        return self.__rank

    def determinant(self):
        return self.__determinant

    def reverse(self):
        return self.__reverseMatrix

    def show(self, show_matrix=False):
        if(show_matrix): self.printMatrix()
        print(f"Degree = {self.__degree}")
        print(f"Determinant = {self.determinant()}")
        print(f"Rank = {self.rank()}")
        print(f"Reverse : \n{self.reverse()}")
        if(self.__degree[0]!=1): print(f"Maggiori zeri : {al.searchZero(self.__matrix)}")
