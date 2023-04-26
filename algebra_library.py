
import numpy as np
import math 
from random import randint

def __preControl(matrix):
    if(not isinstance(matrix, type(np.array([])))): matrix = np.array(matrix)
    
    try:
        matrix.shape[1]
    except IndexError:
        matrix = matrix.reshape((1,-1))
        
    return matrix

def calculateDegree(matrix):
    return tuple((len(matrix),len(np.array(matrix[0]))))

def searchZero(matrix):
    index_row = 0
    row_zero = 0
    for i in range(0, len(matrix)):
        row_zero_temp = 0
        for j in range(0, len(matrix[1])):
            if(matrix[i,j]==0): row_zero_temp += 1
        if(row_zero_temp>row_zero): 
            index_row = i
            row_zero = row_zero_temp
        
    index_col = 0
    col_zero = 0
    for j in range(0, len(matrix[1])):
        col_zero_temp = 0
        for i in range(0, len(matrix)):
            if(matrix[i,j]==0): col_zero_temp += 1
        if(col_zero_temp>col_zero): 
            index_col = j
            col_zero = col_zero_temp

    if(row_zero>=col_zero): return tuple((index_row, 0))
    else: return tuple((index_col, 1))

def reduceMatrix(matrix, row, col):
    new_matrix = np.delete(matrix, row, 0)
    return np.delete(new_matrix, col, 1)

def __sarrus_rule(matrix):
    sarrus_matrix = np.concatenate((matrix, matrix[:, :2]), axis=1)
    result = 0
    for i in range(0, len(matrix[0])):
        y = 0
        value = 1
        for j in range(0,3):
            value *= sarrus_matrix[j,i+y]
            y+=1
        result += value

    for i in range(0, len(matrix[0])):
        y = 0
        value = 1
        for j in range(2, -1, -1):
            value *= sarrus_matrix[j,i+y]
            y+=1
        result -= value

    return result

def isMatrixScale(matrix):
    matrix = __preControl(matrix=matrix)
    for i in range(0, matrix.shape[0]-1):
        for y in range(0, matrix.shape[1]):
            if(matrix[i,y] != 0):
                for j in range(i+1, matrix.shape[0]):
                    for z in range(y, -1, -1):
                        if(matrix[j,z] != 0): return False
                break
    return True

def isMatrixReverse(matrix_a, matrix_b):
    matrix_a = __preControl(matrix=matrix_a)
    matrix_b = __preControl(matrix=matrix_b)

    matrix_i = product_matrix(matrix_a, matrix_b)

    for i in range(0, matrix_i.shape[0]):
        for y in range(0, matrix_i.shape[1]):
            if(i == y and matrix_i[i,y] != 1): return False
            elif(i != y and matrix_i[i,y] != 0): return False
    return True

def calculateDeterminant(matrix, sarrus, historical=False):
    degree = calculateDegree(matrix); determinant_historical = ""
    if(degree[0]!=degree[1]): return None
    elif(degree[0]==1): return matrix[0,0]
    elif(degree[0]==2): return ((matrix[0,0]*matrix[1,1])-(matrix[1,0]*matrix[0,1]))
    elif(degree[0]==3 and sarrus): return __sarrus_rule(matrix)
    else:
        determinant = 0
        major_zeros = searchZero(matrix)
        for i in range(0,len(matrix)):
            if(major_zeros[1]==1):
                reduce_matrix = reduceMatrix(matrix,i,major_zeros[0])
                determinant_historical += f"{matrix[i,major_zeros[0]]} * (-1)^{(i+1)+(major_zeros[0]+1)} * det->\n{reduce_matrix} = {calculateDeterminant(reduce_matrix, sarrus=False)}\n\n"
                determinant += (matrix[i,major_zeros[0]] * (-1)**((i+1)+(major_zeros[0]+1)) * calculateDeterminant(reduce_matrix, sarrus=False))
            else: 
                reduce_matrix = reduceMatrix(matrix,major_zeros[0],i)
                determinant_historical += f"{matrix[major_zeros[0],i]} * (-1)^{(i+1)+(major_zeros[0]+1)} * det->\n{reduce_matrix} = {calculateDeterminant(reduce_matrix, sarrus=False)}\n\n"
                determinant += (matrix[major_zeros[0],i] * (-1)**((i+1)+(major_zeros[0]+1)) * calculateDeterminant(reduce_matrix, sarrus=False))

        determinant_historical = determinant_historical.rstrip(determinant_historical[-2:])
        if(historical): print(determinant_historical)
        return determinant

def __filter_reverse_matrix(reverse_matrix):
    for i in range(0, reverse_matrix.shape[0]):
        for y in range(0, reverse_matrix.shape[1]):
            if(reverse_matrix[i,y] == 0): reverse_matrix[i,y] = abs(reverse_matrix[i,y])
    return reverse_matrix

def calculateReverse(matrix):
    if(matrix.shape[0] != matrix.shape[1]): return None
    determinant = calculateDeterminant(matrix, sarrus=False)
    if(determinant == 0): return None
        
    ac_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    for i in range(0, matrix.shape[0]):
        for y in range(0, matrix.shape[1]):
            ac_matrix[i,y] = (((-1)**(i+y)) * calculateDeterminant(reduceMatrix(matrix, i, y), sarrus=False))
    ac_matrix = ac_matrix.T

    reverse_matrix = (1/determinant) * ac_matrix
    return __filter_reverse_matrix(reverse_matrix)

def __count_zero(row):
    num_zero = 0
    for value in row:
        if(value == 0): num_zero += 1
        else: return num_zero
    return num_zero

def __swap_row(matrix):
    for i in range(0, matrix.shape[0]-1):
        for y in range(i, matrix.shape[0]-1):
            if(__count_zero(matrix[i]) > __count_zero(matrix[y+1])):
                matrix[[i,y+1]] = matrix[[y+1,i]]
    return matrix

def Gauss_Jordan(matrix, show_gaussMatrix=False):
    gauss_matrix = np.array(__preControl(matrix=matrix), dtype=float)
    gauss_matrix = __swap_row(gauss_matrix)
    if(isMatrixScale(matrix=gauss_matrix)): 
        if(show_gaussMatrix): print(f"Gauss_matrix : \n{gauss_matrix}") 
        return matrix
    else:
        for i in range(0, gauss_matrix.shape[0]-1):
            for y in range(0, gauss_matrix.shape[1]):
                if(gauss_matrix[i,y] != 0):
                    for j in range(gauss_matrix.shape[0]-1, i, -1):
                        for z in range(y, -1, -1):
                            if(gauss_matrix[j,z] != 0):
                                scale = -(gauss_matrix[j,z])/gauss_matrix[i,y]
                                gauss_matrix[j] += (gauss_matrix[i]*scale)
                    gauss_matrix = __swap_row(gauss_matrix)
                    if(isMatrixScale(gauss_matrix)): 
                        if(show_gaussMatrix): print(f"Gauss_matrix : \n{gauss_matrix}") 
                        return gauss_matrix
                    break

    if(show_gaussMatrix): print(f"Gauss_matrix : \n{gauss_matrix}") 
    return gauss_matrix

def calculateRank(matrix, show_gaussMatrix=False):
    gauss_matrix = Gauss_Jordan(matrix, show_gaussMatrix=show_gaussMatrix)
    n_pivot = 0
    for i in range(0, gauss_matrix.shape[0]):
        for y in range(0, gauss_matrix.shape[1]):
            if(gauss_matrix[i,y] != 0):
                n_pivot += 1
                break
    return n_pivot

def norm(v):
        v = np.array(v); norm = 0
        for component in v:
            norm += (component**2)
        return math.sqrt(norm)

def getAngle(v1, v2, function='cos'):
    product = np.sum(v1 * v2)
    angle = norm(v1) * norm(v2)
    if(function == 'cos'): angle = round(math.degrees(math.acos(product/angle)), ndigits=5)
    elif(function == 'sin'): angle = round(math.degrees(math.asin(product/angle)), ndigits=5)
    else: return None
    return angle

def product_vector(v1, v2, vectorial=False, get_angle=False):
    v1 = np.array(v1); v2 = np.array(v2)
    if(not vectorial):
        product = np.sum(v1 * v2)
        if(get_angle): return dict({"Product" : product, "Angle": getAngle(v1, v2, function='cos'), "Function": 'cos'})
        else: return product
    else:
        if(v1.shape[0] == 3 and v2.shape[0] == 3):
            m = np.array([v1,v2]).T; v_prod = np.array([])
            for i in range(m.shape[0]-1, -1, -1):
                for y in range(i-1, -1, -1):
                    if(i == 2 and y == 0): matrix_d = np.array([m[i], m[y]])
                    else: matrix_d = np.array([m[y], m[i]])
                    v_prod = np.append(v_prod, calculateDeterminant(matrix_d, sarrus=False))

            if(get_angle): return dict({"Product" : v_prod, "Angle": getAngle(v1, v2, function='sin'), "Function": 'sin'})
            else: return v_prod
        else:
            print("Invalid size vectors!")
            return None

def product_matrix(m_a, m_b):
    if(not isinstance(m_a, type(np.array([])))): m_a = np.array(m_a)
    if(not isinstance(m_b, type(np.array([])))): m_b = np.array(m_b)
    
    try:
        m_a.shape[1]
    except IndexError:
        m_a = m_a.reshape((1,-1))
    
    try:
        m_b.shape[1]
    except IndexError:
        m_b = m_b.reshape((1,-1))

    if(m_a.shape[1] != m_b.shape[0]):
        print(f"Shape error with shape {m_a.shape} != {m_b.shape}")
        return None

    product = np.zeros((m_a.shape[0], m_b.shape[1]))
    for i in range(0, m_a.shape[0]):
        for y in range(0, m_b.shape[1]):
            product[i,y] = np.sum(m_a[i]*m_b[:, y])

    return product

def __verify_linearity(matrix_v, num_vector, e_vector):
    matrix_v_temp = np.array(matrix_v); index_array = list([])
    while(len(index_array) != matrix_v.shape[0] - num_vector):
        random_index = randint(0, matrix_v_temp.shape[0]-1)
        if(random_index not in index_array): index_array.append(random_index)
        if(index_array in e_vector):  index_array = list([])
    matrix_v_temp = np.delete(matrix_v_temp, index_array, axis=0)

    rank = calculateRank(matrix_v_temp, show_gaussMatrix=False)
    if(rank == matrix_v_temp.shape[0]): 
        ind_vectors = list(); dip_vectors = list()
        for i in range(0, matrix_v.shape[0]):
            if(i in index_array): dip_vectors.append(list(matrix_v[i]))
            else: ind_vectors.append(list(matrix_v[i]))
        return dict({"Indipendent-vectors": ind_vectors, "Dipendent-vectors": dip_vectors, "Subspace-Spanned": f"R^{rank}"})
    else:
        e_vector.append(index_array)
        return __verify_linearity(matrix_v, num_vector=num_vector, e_vector=e_vector)


def verify_linearity(*args):
    for vector in args:
        vector = np.array(vector)

    matrix_v = np.array(args, dtype=float)
    rank = calculateRank(matrix_v, show_gaussMatrix=False)
    if(rank != matrix_v.shape[0]): return __verify_linearity(matrix_v, rank, list([]))
    else: return dict({"Indipendent-vectors": args, "Dipendent-vectors": [], "Subspace-Spanned": f"R^{rank}"})
    