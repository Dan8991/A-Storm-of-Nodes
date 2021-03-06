import numpy as np
import scipy as sp
import pandas as pd
from random import uniform
from scipy.sparse.linalg import eigs, eigsh

'''
arr = np array from which values are removed
relative_to = if this value is not None then the values from arr are removed where relative_to == 0
return = arr after removing 0 entries
'''
def remove_zeros(arr, relative_to = None):
    
    #type checking
    assert type(arr) == np.ndarray
    assert (relative_to is None) or (type(relative_to) == np.ndarray)

    if relative_to is None:
        relative_to = arr

    return arr[relative_to != 0]

'''
A = adjacency matrix can be csr_matrix from the scipy module(sparse matrices) or it can be a np matrix
a little test showed that the sparse matrix is 1000 times faster so i suggest using that one
direction = can be either "in" or "out" used to get indegree or outdegree
return = indegree or outdegree of A
'''
def get_degrees(A, direction = "out"):

    if direction == "out":
        axis = 0
    elif direction == "in":
        axis = 1
    else:
        raise Exception("Unsupported degree direction")

    if type(A) == sp.sparse.csr.csr_matrix:
        degrees = np.array(A.sum(axis = axis))
    elif type(A) == np.ndarray:
        degrees = np.sum(A, axis = axis)
    else:
        raise Exception("Unsupported matrix format")
    
    #returning column vector
    return degrees.reshape([max(degrees.shape), 1])

'''
degrees = numpy array containing the indegrees or outdegrees of each node
return = the PDF of the degrees 
'''
def get_neighbours_pdf(degrees):

    assert type(degrees) == np.ndarray

    #removing nodes with no connections
    degrees = remove_zeros(degrees)
    
    #getting the degrees of the nodes and the number of occurrences of each value
    unique, counts = np.unique(degrees, return_counts = True)
    
    #calculating the pdf
    pdf = counts / np.sum(counts)

    return unique, pdf

'''
pdf = probability distribution function over which the ccdf is calculated, has to be a np array
return = the ccdf associated to pdf
'''
def ccdf(pdf):

    assert type(pdf) == np.ndarray

    return 1 - np.cumsum(pdf)

'''
degrees = array conatining the indegrees or outdegrees of each node
return = the estimated exponent of the power law and the constant c in front of the expression
'''
def estimate_power_law(degrees, kmin):

    assert (type(degrees) == np.ndarray) and (type(kmin) == int) and (kmin > 0)
    

    #removing unnecessary degrees
    degrees = degrees[degrees > kmin]
    
    #calculating the parameters of the power law
    gamma = 1 + len(degrees)/np.sum(np.log(degrees / kmin))
    c = (gamma - 1)*kmin**(gamma - 1)
    
    return gamma, c

'''
exp_step_size = how much the exponent of the x is increased at every step
degrees = degrees of the node of the network
return = the bins and the probability associated with each bin
'''
def log_binning_pdf(exp_step_size, degrees):
    
    assert (exp_step_size > 0) and (type(degrees) == np.ndarray)

    degrees = remove_zeros(degrees)

    #getting the x bins
    x = 10 ** np.arange(0, np.ceil(np.log10(np.max(degrees))), exp_step_size)
    
    #getting the probability of being in a bin
    log_bin_pdf, _ = np.histogram(degrees, x)
    log_bin_pdf = log_bin_pdf / np.sum(log_bin_pdf)

    return [x[:-1], log_bin_pdf]

'''
m = number of returned random values
p = probability distribution
val_p = values associated with the probabilities, in val_p == None the they are set to the numbers from 1 to len(p)
return = m values extracted from val_p given the pdf
'''
def random_connections(m, p, val_p = None):
    
    #some basic checks on the inputs
    assert (type(m) == int) and (m > 0)
    assert (type(p) == np.ndarray)
    assert (val_p is None) or ((type(val_p) == np.ndarray) and (val_p.shape == p.shape))

    #setting val_p to an array from 0 to the length of p if it is not already defined
    if val_p == None:
        val_p = np.arange(len(p))

    #getting the values accordingly to the pdf using the cdf method
    rand = np.random.rand(m)
    cdf = np.cumsum(p)
    values = np.digitize(rand, cdf)

    return values
    

'''
N = dimension of the A matrix
m = amount of new links at any iteration
At = preferential attachment
fitness = array of fitnesses, it is supposed to be a Nx1 np array
return = csr_matrix(sparse) generated with the barabasi albert model
NOTES: the implementation is slightly different from the one presented in
matlab because this way in python it is faster.
'''
def get_barabasi_albert_net(N, m, At = 0, fitness = None):
    
    #some basic checks on the inputs used to avoid common errors
    assert (type(N) == int) and (N > 0)
    assert (type(m) == int) and (m > 0)
    assert (type(At) == int) and (At >= 0)
    assert (fitness is None) or ((type(fitness) == np.ndarray) and (len(fitness) == N))

    #if no fitness is specified then it is set as equal for all the nodes
    if fitness is None:
        fitness = np.ones((N, 1))

    #since we need to perform assignment in this case np.array is faster than sparse
    A = np.zeros((N, N))
    A[0, 0] = 1
    degrees = np.ones((1, 1))
    
    for k in range(1, N):

        #adding one connection to the new node in order to allow self loops
        degrees = np.append(degrees, 1)

        pa_degrees = degrees + At

        fit_degrees = pa_degrees * fitness[np.arange(k + 1)].T
        #normalizing
        p_connect = fit_degrees/np.sum(fit_degrees) 

        #getting the links
        links = random_connections(m, p_connect)

        #counting links between two node only once
        unique_links = np.unique(links)
 
        #increasing the degree of every node connected with k
        degrees[unique_links] += 1

        #setting the degree of node k equal to the number of links it is connected to
        degrees[k] = len(unique_links)

        #updating the matrix
        A[links, k] = 1
        A[k, links] = 1
    
    #returning csr_matrix because it is faster at doing most calculations
    return sp.sparse.csr_matrix(A)

'''
A = sparse matrix
return = sparse matrix that is undirected, this is done by adding an edge in the opposite direction 
where there is already one
'''
def get_undirected_network(A):
    
    #type checking
    assert type(A) == sp.sparse.csr_matrix

    return 1*((A + A.T) > 0)  

'''
A = sparse matrix in particular a csr_matrix from scipy.sparse
return = cleaned sparse csr matrix containing only nodes from the Giant Component
'''
def clean_network(A):

    assert type(A) == sp.sparse.csr_matrix

    #building an undirected network
    Au = get_undirected_network(A)
    N = A.shape[0]
    
    #isolating the GC

    #non visited nodes will have value 1
    not_visited = np.ones((N, 1))

    #size of the biggest component found until now
    biggest_component = 0
    best = None

    while np.sum(not_visited) > biggest_component:

        #get first non zero index
        index = np.where(not_visited)[0][0]
        
        #getting the distance from the node to all other nodes 
        #non reached nodes have distance -1
        distances = breadth_first_search(Au, int(index))
        connected = distances >= 0

        # #setting all visited nodes = 0
        not_visited = not_visited - connected
        
        #select the best bigger component
        if np.sum(connected) > biggest_component:
            best = connected
            biggest_component = np.sum(connected)

    e1 = np.reshape(best, (N))

    #this is apparently the most efficient way of slicing a sparse matrix
    A = A[e1, :][:, e1]

    return A

'''
A = sparse csr_matrix
starting_node = node from which you want to start BFS 
return = array of distances between node i and the starting node, 
all distances to unreachable nodes are set to -1
'''
def breadth_first_search(A, starting_node):
    
    #type checking
    assert type(A) == sp.sparse.csr.csr_matrix
    N = A.shape[0]
    assert type(starting_node) == int and starting_node < N
    
    e1 = np.zeros((N,1))
    
    #setting to 1 the value corresponding to the starting node
    e1[starting_node] = 1

    #all distances are set to -1 except from the one corresponding to the starting
    #node that is obviously 0
    distances = -np.ones((N, 1))
    distances[starting_node] = 0

    #value counting the distance from the starting node
    dist = 0
    
    #exit condition
    ex = False

    while not ex:

        #distance is increased at every step
        dist += 1

        e1_old = e1
        
        #searching for nodes connected to the nodes in e1
        e1 = (A * e1 + e1) > 0

        #finidng out new nodes
        new_nodes = e1 != e1_old

        #setting the distances of the new nodes to the starting node
        distances[new_nodes] = dist

        #checking if no new nodes were added to the list
        ex = not np.sum(new_nodes)

    return distances

'''
A = sparse matrix whose distance distribution is calculated
return = unique distances and how many time each distance is found
'''
def get_distance_distribution(A):
    
    #type checking
    assert type(A) == sp.sparse.csr.csr_matrix

    N = A.shape[0]

    #empty array to be later concatenated
    distances = np.zeros((0,1))
    
    #getting distances for all nodes
    for i in range(N - 1):
        dist = breadth_first_search(A, i)

        #only adding distances to nodes after the considered one
        #because the previous ones were already calcolated
        distances = np.concatenate([distances, dist[(i + 1):]], axis=0)
    
    return np.unique(distances, return_counts = True)


'''
A = density matrix as a csr_matrix
neigh_dir = direction in which to calculate the degree of the nodes can be "in" or "out"
knn_dir = direction in which the average neighbours degree is calculated can be "in" or "out"
return = [p, unique, knn, temp_knn] coefficients of the polinomial fitting
(the one at index 0 is the assortativity value), unique degrees and knn
'''
def get_assortativity_value(A, neigh_dir = "out", knn_dir = "out"):

    #type checking
    assert type(A) == sp.sparse.csr_matrix
    assert (neigh_dir in ["in", "out"]) and (knn_dir in ["in", "out"])
    
    #getting the degrees
    neigh_deg = get_degrees(A, neigh_dir)
    knn_deg = get_degrees(A, knn_dir)

    if neigh_dir == "out":
        A = A.T

    #done in order to evade division by zero, non meaningful results are removed later
    temp_neigh_deg = np.copy(neigh_deg)
    temp_neigh_deg[temp_neigh_deg == 0] = 1

    temp_knn = (A * sp.sparse.csr_matrix(knn_deg))/temp_neigh_deg
    
    #removing unmeaningful values
    temp_knn = remove_zeros(np.array(temp_knn), relative_to = neigh_deg)
    neigh_deg = remove_zeros(neigh_deg)

    #setting temp_knn as a column vector
    temp_knn = np.reshape(temp_knn, [max(temp_knn.shape), 1])

    #removing the 0 values
    unique = np.unique(neigh_deg)
    
    #df_data is needed to create a pandas DataFrame that will make next calculation way easier
    df_data = np.concatenate([neigh_deg[:, np.newaxis], temp_knn], axis = 1)

    #getting knn
    knn = pd.DataFrame(df_data, columns = ["degrees", "temp_knn"]).groupby("degrees").mean().values

    #removing 0 values that would go to infinity in the log
    unique = remove_zeros(unique[:, np.newaxis], relative_to = knn)
    knn = remove_zeros(knn)

    #fitting the data with a linear model
    p = sp.polyfit(np.log(unique), np.log(knn), 1)

    #returning the slope of the linear model
    return p, unique, knn, temp_knn

'''
A = sparse matrix from which the clustering coeffitients have to be calculated
node = nod whose clustering coefficient is calculated
return = clustering coefficient of node
'''

def get_clusteing_coefficient(A, node):
    
    assert type(A) == sp.sparse.csr.csr_matrix
    N = A.shape[0]
    assert type(node) == int and node < N

    e = np.zeros((N, 1))
    e[node] = 1

    #finding the neighbours of node and their number
    neighbours = A * e
    neigh_num = np.sum(neighbours)

    #reshape needed for slicing
    neighbours = np.reshape(neighbours, [N]) == 1

    #getting the subgraph contanining only the neighbours of node
    neigh_subgraph = A[neighbours, :][:, neighbours]
    
    #finding the number of connections between the neighbours
    neigh_conn = np.sum(neigh_subgraph) / 2

    #maximum number of connections between the neighbours
    conn_max = neigh_num * (neigh_num - 1) / 2

    #handles the case when there is only one neighbour
    #in this case the clustering coefficient is 0
    if conn_max == 0:
        return 0

    return neigh_conn/conn_max

'''
A = sparse matrix from which the clustering coeffitients have to be calculated
return = clustering coefficients
'''
def get_clustering_coefficients(A):
    
    assert type(A) == sp.sparse.csr.csr_matrix
    N = A.shape[0]
    
    return np.array([get_clusteing_coefficient(A, i) for i in range(N)])

'''
A = sparse matrix from which the clustering distribution has to be calculated
return = clustering coefficients unique values and PDF
'''
def get_clustering_distribution(A):
    return get_neighbours_pdf(get_clustering_coefficients(A))

'''
N = nodes in the network
nodes = array of arrays of connections, one array of connections for each character
return = matrix where each column is the temporal evolution of a character
'''
def get_temporal_distribution(N, nodes):

    assert type(N) == int
    assert type(nodes) == list

    #initializing matrix and return value
    A = sp.sparse.csr_matrix((N, N))
    temporal = []

    #iterating through each chapter
    for chap in nodes:

        #check needed because some chapters might be empty
        if len(chap) > 0:

            edges = np.array(chap)

            #converting sparse matrix to a type where assignment is faster
            A = A.tolil()

            #adding the connections formed in the chapter
            A[edges[:, 0], edges[:, 1]] = 1
            A[edges[:, 1], edges[:, 0]] = 1

            #converting the matrix back to a type where operations are faster
            A = A.tocsr()

            #getting degrees and adding them to the return list
            degrees = get_degrees(A)
            temporal.append(degrees)
    
    return np.array(temporal)

'''
A = sparse matrix used for the robustness check
return = array of GC dimensions and # of nodes removed before breakdown both for random and hubs removal
'''
def check_robustness(A):

    assert type(A) == sp.sparse.csr_matrix

    return random_node_removal(A), attack_node_removal(A)

'''
A = sparse matrix where hub removal is performed
return = array of GC dimensions and # of removal needed to break the GC
'''
def random_node_removal(A):

    assert type(A) == sp.sparse.csr_matrix

    #copying the matrix in order to avoid side effects
    A1 = sp.sparse.csr_matrix(A, copy = True)

    #getting a random array as the order of removal
    N = A1.shape[0]
    remove_order = np.random.permutation(N)

    return get_gc_dimension_removing_nodes(A1, remove_order)

'''
A = sparse matrix where hub removal is performed
return = array of GC dimensions and # of removal needed to break the GC
'''
def attack_node_removal(A): 
    
    assert type(A) == sp.sparse.csr_matrix
    
    #copying the matrix in order to avoid side effects
    A1 = sp.sparse.csr_matrix(A, copy = True)

    #getting the degrees in order to know which are the hubs
    degrees = get_degrees(A1)

    #getting the indexes that would sort the array
    remove_order = np.argsort(degrees, axis = 0)

    #getting the indexes of the hubs at the beginning of the array
    remove_order = remove_order[::-1]

    return get_gc_dimension_removing_nodes(A1, remove_order)


'''
degrees = array of degrees of the nodes in the network
return = inhomogeneity value
'''
def get_inhomogeneity(degrees):

    #calculating mean degree
    mean_degree = np.mean(degrees)
    
    #this happens if there isn't a network anymore
    if mean_degree == 0:
        return 0

    return sp.stats.moment(degrees, moment=2)/mean_degree


'''
A = sparse matrix from where the nodes have to be removed
remove_order = array of indexes, nodes are removed accordingly to theese indexes
return:
gc_dimension = array of dimensions of the GC after every removal
inhomogeneity_break = # of removal needed for the GC to dissapear
'''
def get_gc_dimension_removing_nodes(A, remove_order):
    
    assert type(A) == sp.sparse.csr_matrix
    assert type(remove_order) == list or type(remove_order) == np.ndarray

    gc_dimension = []
    inhomogeneity_break = 0

    #removing nodes based on remove_order
    for remove in remove_order:

        #removing node
        A[remove, :] = 0
        A[:, remove] = 0

        #calculating and saving the current GC dimension
        gc_dimension.append(clean_network(A).shape[0])

        #calculating inhomogeneity ratio and updating the counting variable while it is > 2
        if get_inhomogeneity(get_degrees(A)) > 2:
            inhomogeneity_break += 1
    
    return gc_dimension, inhomogeneity_break

'''
A = sparse matrix whose links have to be rewired
return = matrix generated with the Molloy Reed algorithm.
'''
def random_rewiring(A):
    
    assert type(A) == sp.sparse.csr_matrix
    
    N = A.shape[0]

    degrees = get_degrees(A).reshape(N)

    #array of numbers that goes from 0 to N
    indexes = np.arange(N)

    #permuting an array that contains and index i d_i times, where d_i is the degree of node i
    connections = np.random.permutation(np.repeat(indexes, degrees))

    #x = index of first node index of the second node to be connected, y = index of second node to be connected
    x = np.concatenate([connections[0::2], connections[1::2]])
    y = np.concatenate([connections[1::2], connections[0::2]])
    
    #building up the adjacency matrix with the previously created random connections
    A_rand = sp.sparse.csr_matrix((np.ones(len(x)), (x, y)), shape=(N,N), dtype = np.int32)

    return A_rand

'''
A = sparse matrix representing the network
c = damping factor
q = teleport vector
return = page rank ranking vector
'''
def page_rank_linear_system(A, c = 0.85, q = None):
    
    assert type(A) == sp.sparse.csr.csr_matrix
    assert (type(c) == float) and (c > 0) and (c < 1)
    assert (q is None) or type(q) == np.ndarray

    N = A.shape[0]
 
    #if no q is passed then a q with all probabilities equal to 1/N is created
    if q is None:
        q = np.ones((N, 1))/N

    d = 1/get_degrees(A)
    M = A * sp.sparse.diags(d[:, 0])

    #finding a and b such that ap = b
    a = (np.eye(N) - c * M)/(1-c)
    
    #this is faster than the sparse solver because p is not a sparse vector 
    #since a has at leas one positive entry for each row/column and b doesn't have any 0
    p = np.linalg.solve(a, q)
    p = p/np.sum(p)

    return p

'''
A = sparse matrix
return = sparse matrix where there are no single nodes and all dead ends are removed
'''
def remove_dead_ends(A):
    
    assert type(A) == sp.sparse.csr_matrix

    #exit condition
    ex = False

    while not ex:
        #if the outdegree of the node is 0 remove it
        pos = (get_degrees(A) != 0).reshape(-1)
        A = A[pos, :][:, pos]

        #checking if there are any bad nodes left
        ex = np.sum(get_degrees(A) == 0) == 0
    
    return A

'''
A = sparse matrix whose page rank is calculated
iter_num = number of iterations
c = damping factor
q = teleport vector
p_linear = real page ranking obtained with linear system solution
'''
def page_rank_power_iteration(A, iter_num = 35, c = 0.85, q = None, p_linear=None):
    
    #type checks
    assert type(A) == sp.sparse.csr.csr_matrix
    assert (type(c) == float) and (c > 0) and (c < 1)
    assert (q is None) or (type(q) == np.ndarray)
    assert (type(iter_num) == int) and (iter_num > 0)
    assert (p_linear is None) or (type(p_linear) == np.ndarray)

    N = A.shape[0]
 
    #if no q is passed then a q with all probabilities equal to 1/N is created
    if q is None:
        q = np.ones((N, 1))/N
    
    #ranking starting point, all nodes have the same rank
    pt = np.ones((N, 1))/N

    #calculating the M matrix
    d = 1/(get_degrees(A) + 1e-10) 
    M = A * sp.sparse.diags(d[:, 0])
    
    errors = []

    for _ in range(iter_num):

        #updating and normalizing the ranking
        pt = c * M * pt + (1 - c) * q
        pt = pt/np.sum(pt)
        
        #if the real rank was passed the error is computed
        if p_linear is not None:
            errors.append(np.linalg.norm(p_linear - pt)/ N ** 0.5)

    #returning the error if it was calculated
    if p_linear is not None:
        return pt, errors
    
    return pt

'''
A = sparse matrix
return = the 2 biggest eigenvalues of the matrix A
'''
def get_two_highest_eigenvalues(A):
    
    assert type(A) == sp.sparse.csr_matrix

    #needed for the sparse function to work
    A = A.astype(float)

    #getting the highes eigenvalues
    val, _ = eigs(A, 2)

    return val

'''
A = sparse matrix
return = HITS rank of A obtained by finding the eigenvector relative to the second highest eigenvalue
'''
def hits_linear_system(A):

    assert type(A) == sp.sparse.csr_matrix

    #getting the M matrix used in HITS
    M = A * A.T

    #getting the highest eigenvalues and the relative eigenvectors
    M = M.astype(float)
    _, vec = eigs(M, k = 2)
    
    #normalizing the eigenvector
    p = -vec[:, 0]/np.linalg.norm(vec[:, 0])

    #the abs is returned because the vector is complex with imaginary part 0
    #his is not a problem because p is a probability vector so all entries must be positive
    return np.abs(p)

'''
A = sparse matrix
iter_num = number of iterations
p_linear = real HITS rank
return = HITS rank found thanks to power iteration
'''
def hits_power_iteration(A, iter_num = 35, p_linear=None):
    
    assert type(A) == sp.sparse.csr.csr_matrix
    assert (type(iter_num) == int) and (iter_num > 0)
    assert (p_linear is None) or (type(p_linear) == np.ndarray)

    N = A.shape[0]
    
    #giving the same rank to every node
    pt = np.ones((N, 1))/N**0.5

    #finding the M matrix
    M = A*A.T
    
    errors = []

    for _ in range(iter_num):

        #updating pt
        pt = M*pt
        #normalizing pt
        pt /= np.linalg.norm(pt)

        #if the true pt is provided the error is calculated
        if p_linear is not None:
            errors.append(np.linalg.norm(p_linear - pt.T)/ N ** 0.5)

    if p_linear is not None:
        return pt, errors
    
    return pt

'''
A = sparse matrix
return = normalized laplacian of A
'''
def get_normalized_laplacian(A):
    
    assert type(A) == sp.sparse.csr_matrix

    N = A.shape[0]

    D = get_D_matrix(A)
    L = sp.sparse.identity(N) - D*A*D

    return L

'''
A = sparse matrix
return = D sparse matrix give A
'''
def get_D_matrix(A):

    assert type(A) == sp.sparse.csr_matrix
    
    N = A.shape[0]

    d = get_degrees(A)
    inv_d = np.reshape(1/np.sqrt(d), N)
    D = sp.sparse.diags(inv_d.T, offsets=0)

    return D

'''
A = sparse matrix
reorder = vector according to whom A should be reordered
inverse = if False reorder is sorted from smaller to bigger if true the opposite is done
reuturn = A reordered according to reorder from it's smaller value to the biggest
'''
def reorder_nodes(A, reorder, inverse=False):

    assert type(A) == sp.sparse.csr_matrix
    
    N = A.shape[0]
    
    assert type(reorder) == np.ndarray
    assert type(inverse) == bool

    reorder = reorder.reshape(N)
    ids = np.argsort(reorder)
    
    if inverse:
        ids = ids[::-1]

    A1 = A[ids, :][:, ids]

    return A1, ids

'''
A = sparse matrix
return = fiedler vector and it's successor
'''
def get_fiedler_vector(A):

    assert type(A) == sp.sparse.csr_matrix

    N = A.shape[0]
    
    D = get_D_matrix(A)
    L = sp.sparse.identity(N) - D*A*D

    #getting the eigenvectors of L
    _, eig_vec = eigsh(L, k = 3, which="SM")
    
    #normalizing the eigenvectors
    eig_vec = D * eig_vec

    return eig_vec[:, 1], eig_vec[:, 2]

'''
A = sparse matrix
return = conductance array of matrix A
'''
def get_conductance(A):

    assert type(A) == sp.sparse.csr_matrix

    N = A.shape[0]

    #values needed1 to calculate cut and assoc
    a = np.asarray(sp.sparse.triu(A).sum(axis=0))
    b = np.asarray(sp.sparse.tril(A).sum(axis=0))
    d = get_degrees(A)

    cut = np.cumsum(b - a, axis = 1, dtype=np.float32)

    assoc = np.cumsum(d, axis = 0, dtype = np.float32)

    D = np.sum(d)

    denominator =  np.min(np.concatenate([assoc, D - assoc], axis = 1), axis = 1)

    #changing 0 to very small values
    cut[cut == 0] = 1e-10
    denominator[denominator == 0] = 1e-10

    conductance = cut.T.reshape(N) / denominator

    return conductance.T

'''
A = sparse matrix
epsilon = precision
starting_node = starting node
c = damping factor
return = approximate page nibble
'''
def page_nibble_with_finite_precision(A, epsilon=1e-3, starting_node = 0, c = 0.85):

    assert type(A) == sp.sparse.csr_matrix
    assert type(epsilon) == float

    N = A.shape[0]

    assert (type(starting_node) == int) and (starting_node < N)

    #getting basic parameters to perform the page nibble
    d = get_degrees(A)
    M = A * sp.sparse.diags(1/(d[:, 0]))
    D = np.sum(d)
    q = np.zeros((N,1))
    q[starting_node] = 1
    u = np.zeros((N, 1))
    v = q.copy()
    th = epsilon * d / D

    #while some values of v are bigger than the threshold 
    while np.sum(v > th) > 0:

        #compute the delta where the vector is bigger than threshold
        delta = v.copy()
        delta[v < th] = 0

        #updating u,v
        u += (1-c)*delta
        v = v - delta + c*M*delta
        
    return u

'''
divides a network in communities via bisection
A = sparse matrix of the network
function = f(A) given A it returns the conductance and the indexes used for reordering
conductance_lim = maximum conductance, if conductance is greater than this the community is not split
return = list of communities and list of separators
'''
def divide_in_communities(A, function , conductance_lim = 0.3):

    assert type(A) == sp.sparse.csr_matrix
    assert (type(conductance_lim) == float) and (conductance_lim <= 1) 

    indexes = np.arange(A.shape[0])

    return recursive_communities(A, indexes, conductance_lim, function)

'''
A = matrix to be divided in communities
indexes = indexes of the original nodes that are in the community
conductance_lim = maximum conductance value to get a bisection
function = f(A) given A it returns the conductance and the indexes used for reordering
path = string used to return the dendrogram 0 means that community is in the left subtree, 1 on the right one
border = separating node
return = list of communities and list of separators
'''
def recursive_communities(A, indexes, conductance_lim, function, path="", border = -1):
    
    assert type(A) == sp.sparse.csr_matrix
    N = A.shape[0]

    assert type(indexes) == np.ndarray
    assert (type(conductance_lim) == float) and (conductance_lim <= 1) 
    assert type(path) == str
    assert type(border) == int

    #getting conductance and ids from function
    conductance, ids = function(A)

    #if the minimum conductance is bigger than the maximum limit than no division is performed   
    if np.min(conductance) > conductance_lim:
        return [{"path":path, "indexes":indexes, "border":border}], []

    separator = np.argmin(conductance) + 1

    #if one of the two communities has less than 3 nodes no split is performed
    if (separator < 4) or (separator > N-4):
        return [{"path":path, "indexes":indexes, "border":border}], []   
    
    #getting the communities defined by ids and the separators
    C1, C2, A1, A2, communities = get_communities(A, ids, separator, indexes)

    #getting the modularity of the network given the split
    modularity_divided = get_modularity(A, communities)

    #if modularity decreases no split is performed
    if modularity_divided < 0:
        return [{"path":path, "indexes":indexes, "border":border}], []

    #recursive part, the two communities are split again
    div1, separator1 = recursive_communities(A1, C1, conductance_lim, function, path + "0", int(indexes[ids[separator]]))
    div2, separator2 = recursive_communities(A2, C2, conductance_lim, function, path + "1", int(indexes[ids[separator]]))

    #generating the set of separators
    separator_set = []
    if len(separator1) > 0:
        separator_set.append(separator1)
    separator_set.append(indexes[ids[separator-1]])
    if len(separator2) > 0:
        separator_set.append(separator2)

    #returning the array of splits and of separators
    return div1 + div2, separator_set

'''
A = sparse matrix representing the network
ids = indexes that sort the nodes so that the computed conductance is correct
separator = index of the minimum conductance
indexes = real indexes of the nodes of the community
return:
C1 = real indexes of nodes in community one
C2 = real indexes of nodes in community two
A1_full = matrix of the first community
A2_full = matrix of the second community
communities = binary array that tells which of the nodes in indexes belong to each community
'''
def get_communities(A, ids, separator, indexes):
    
    assert type(A) == sp.sparse.csr_matrix 
    assert type(ids) == np.ndarray
    assert (type(separator) == int) or (type(separator) == np.int64) or (type(separator) == np.int32)
    assert type(indexes) == np.ndarray

    #dividing the two communities
    C1_ids = ids[:separator]
    C2_ids = ids[separator:]

    #getting the two matrices representing the communities
    A1 = A[C1_ids,:][:, C1_ids]
    A2 = A[C2_ids,:][:, C2_ids]

    #getting the degrees of the nodes in the communities
    d1 = get_degrees(A1).reshape(-1)
    d2 = get_degrees(A2).reshape(-1)

    '''
    this basically takes nodes with degree 0 in one communitiy and moves them to the others 
    this is done because initially no node had 0 degree so if one has it means that it was moved there 
    by chance but the node should absolutely be part of the community where he has degree != 0
    '''
    C1_ids_full = np.concatenate([C1_ids[d1 != 0].reshape(-1, 1), C2_ids[d2==0].reshape(-1, 1)], axis = 0).reshape(-1)
    C2_ids_full = np.concatenate([C1_ids[d1 == 0].reshape(-1, 1), C2_ids[d2!=0].reshape(-1, 1)], axis = 0).reshape(-1)

    #getting the matrices based on the new communities
    A1_full = A[C1_ids_full,:][:, C1_ids_full]
    A2_full = A[C2_ids_full,:][:, C2_ids_full]

    #getting the real communities with correct indexes
    C1 = indexes[C1_ids_full.reshape(-1)]
    C2 = indexes[C2_ids_full.reshape(-1)]

    #returns a binary vector that distinguishes between the two communities
    communities = np.zeros(A.shape[0])
    communities[C2_ids_full] = 1

    return C1, C2, A1_full, A2_full, communities

'''
A = sparse matrix of the network
return: 
conductance = conductance calculated through spectral clustering reordering, 
ids = indexes reordering based on fiedler's vector
'''
def spectral_clustering_reordering(A):

    assert type(A) == sp.sparse.csr_matrix

    #getting fiedler's vector
    v1, _ = get_fiedler_vector(A)

    #reordering matrix A according to fiedler's vector
    A1, ids = reorder_nodes(A, v1)

    #getting the conductance
    conductance = get_conductance(A1)

    return conductance, ids

'''
A = sparse matrix of the network
return:
conductance = conductance calculated through page nibble reordering, 
ids = indexes reordering based on page nibble
'''
def page_nibble_split(A):

    assert type(A) == sp.sparse.csr_matrix

    N = A.shape[0]
    
    #choosing sqrt(N) random initials node where to start from, only the best split will be considered
    choices = np.random.choice(N, size = int(np.ceil(np.sqrt(N))), replace=False)
    
    #setting the variables used to find out the best split
    best_conductance = 0
    best_ids = 0
    min_conductance = 2

    #for each choiche 
    for choice in choices:

        #perform page nibble starting from choice
        q = page_nibble_with_finite_precision(A, starting_node=int(choice))

        #reordering nodes
        A1, ids = reorder_nodes(A, q)

        #calculating the conductance
        conductance = get_conductance(A1)

        #finding the minimum conductance value
        cmin = np.min(conductance)

        #if this minimum is smaller than the best one until now update everything
        if cmin < min_conductance:
            min_conductance = cmin
            best_conductance = conductance
            best_ids = ids

    #return the best results
    return best_conductance, best_ids


'''
A = sparse matrix representing the network
return = similarity matrix with removed values on the diagonal and already existing links
'''
def common_neigh_link_prediction(A):
    
    assert type(A) == sp.sparse.csr_matrix

    #getting the similarity matrix based on the number of common neighbours
    S = (A*A).toarray()

    return clean_link_prediction_matrix(S, A)

'''
A = sparse matrix representing the network
i = first node
j = second node
return = boolean vector highliting common neighbours between i and j
'''
def find_common_neigh(A, i, j):
    
    assert type(A) == np.ndarray
    
    N = A.shape[0]

    assert (type(i) == int) and (i >= 0) and (i<N)
    assert (type(j) == int) and (j >= 0) and (j<N)

    #finding out which neighbours they have in common
    return ((A[i] == A[j])&(A[i] == 1)).reshape(-1)

'''
A = matrix representation of the network
return = similarity matrix with removed values on the diagonal and already existing links
'''
def adamic_adar_link_prediction(A):

    assert type(A) == sp.sparse.csr_matrix

    N = A.shape[0]

    #getting degrees
    d = get_degrees(A).astype(np.float32)

    #making sure that no degree is exactly one otherwise 1/ln(1) = inf
    d[np.isclose(d, 1)] = 1 + 1e-10

    #initializing S
    S = np.zeros((N,N))
    A = A.toarray()

    #for each node
    for i in range(1,N):
        #no need to iterate again through all nodes since the matrix is symmetrical
        for j in range(i):
            
            #finding common neighbours
            common = find_common_neigh(A, i, j)

            #if the two have at least one neighbours
            if np.sum(common) > 0:

                #finding the value of similarity between i and j
                Sij = np.sum(1/np.log(d[common]))
                S[i,j] = Sij
                S[j,i] = Sij

    #removing the diagonal and nodes that are already connected
    return clean_link_prediction_matrix(S, sp.sparse.csr_matrix(A))

'''
A = matrix representation of the network
return = similarity matrix with removed values on the diagonal and already existing links
'''
def resource_allocation_link_prediction(A):
    
    assert type(A) == sp.sparse.csr_matrix
    
    N = A.shape[0]

    #getting degrees and setting up the matrix
    d = get_degrees(A)
    S = np.zeros((N,N))
    A = A.toarray()

    #for each node
    for i in range(1, N):
        #no need to iterate again through all nodes since the matrix is symmetrical
        for j in range(i):

            #getting common neighbours
            common = find_common_neigh(A, i, j)

            #if i and j have at least 1 common neighbour
            if np.sum(common) > 0:

                #calculating similarity
                Sij = np.sum(1/d[common])
                S[i,j] = Sij
                S[j,i] = Sij

    #removing the diagonal and nodes that are already connected
    return clean_link_prediction_matrix(S, sp.sparse.csr_matrix(A))

'''
A = matrix representation of the network
l = maximum exponent
beta = discount factor
return = similarity matrix with removed values on the diagonal and already existing links
'''
def katz_link_prediction(A, l, beta):

    assert type(A) == sp.sparse.csr_matrix
    assert (type(l) == int) and (l >= 2)
    assert type(beta) == float
    
    N = A.shape[0]
    
    #initializing the similarity matrix
    S_katz = sp.sparse.csr_matrix((N,N))
    
    #starting from A**2 up to A**l, the bigger l the less sparse the matrix meaning that it takes more time
    for i in range(2, l+1):

        #updating similarity matrix
        S_katz += A**i*beta**i

    S = S_katz.toarray()

    #removing the diagonal and nodes that are already connected
    return clean_link_prediction_matrix(S, A)

'''
computes Area Under the Receiver Operating Characteristic Curve given a function f that calculates nodes similarity
A = sparse matrix representing a network
f = function that calculates similarity between nodes, it is of type f(A, *args)
args = additional arguments of function f as a python list
return = AUC_ROC of the predicted links considering real links
'''
def ROC_AUC(A, f, args=None):

    assert type(A) == sp.sparse.csr_matrix
    assert (args is None) or (type(args) == list) or (type(args) == tuple)

    N = A.shape[0]

    #findig out where there is already a link between nodes
    x,y = np.where(sp.sparse.triu(A).toarray() == 1)

    #finding out how many link choiches there are
    possible_choiches = np.arange(len(x))

    #getting one tenth of the overall links as probe set
    choiches = np.random.choice(possible_choiches, size=(x.shape[0] // 10), replace=False)
    p = np.concatenate([x[choiches].reshape(-1, 1), y[choiches].reshape(-1, 1)], axis = 1).T
    values = np.ones((p.shape[1]))

    #building the probe matrix
    A_p = sp.sparse.csr_matrix((values, p), shape = (N, N))
    A_p = A_p + A_p.T

    #getting the test set matrix
    A_t = A - A_p

    #getting the similarity matrix from the test set
    if args is not None:
        S_t = f(A_t, *args)
    else:
        S_t = f(A_t)

    #getting a boolean matrix of inactive nodes
    A_i = np.triu(A.toarray() != 1)

    #getting a boolean matrix of the probe set
    A_p = np.triu(A_p.toarray() == 1)

    #getting the predicted values in the probe set
    p = S_t[A_p].reshape(-1,1)

    #getting the predicted values in the test set
    i = S_t[A_i].reshape(-1,1)

    #finding out how many times the prediction from the probe set is bigger than a prediction from inactive set
    numerator = np.sum(i < p.T)
    
    #returning the normalized result
    return numerator/i.shape[0]/p.shape[0]


'''
A = matrix representation of the network
return = similarity matrix with removed values on the diagonal and already existing links
'''
def random_walk_with_restart_link_prediction(A):

    assert type(A) == sp.sparse.csr_matrix

    N = A.shape[0]
    ranking = np.zeros((N, N))
    
    #for each node in the network perform page rank with node i as teleport vector
    for i in range(N):

        #defining teleport vector
        q = np.zeros((N,1))
        q[i] = 1
        
        #calculating page rank
        pt = page_rank_power_iteration(A, q=q)
        
        #calculating a piece of the similarity matrix
        ranking[i, :] = pt.reshape(1,-1)

    #getting the similarity matrix
    S = ranking + ranking.T

    #removing the diagonal and nodes that are already connected
    return clean_link_prediction_matrix(S, A)

'''
A = matrix representation of the network
t = number of steps
return = similarity matrix with removed values on the diagonal and already existing links
'''
def local_random_walk_link_prediction(A, t=5):

    assert type(A) == sp.sparse.csr_matrix
    assert (type(t) == int) and (t > 0) 

    N = A.shape[0]

    #getting the M matrix
    d = get_degrees(A)
    #some nodes might have degree 0 since some links can be removed by ROC_AUC
    M = A * sp.sparse.diags((1/(d + 1e-10))[:, 0])
    #performing the random walk
    Mt = M**t

    #for each i,j calculating pij*d_i + pji*d_j
    Mt = Mt*(np.zeros((1, N)) + d)
    S = Mt + Mt.T

    #removing the diagonal and nodes that are already connected
    return clean_link_prediction_matrix(S, A)

'''
A = matrix representation of the network
t = number of steps
return = similarity matrix with removed values on the diagonal and already existing links
'''
def superposed_random_walk_link_prediction(A, t=5):

    assert type(A) == sp.sparse.csr_matrix
    assert (type(t) == int) and (t > 0)

    N = A.shape[0]

    S = np.zeros((N, N))

    #for each timestep from 1 to t
    for u in range(1, t + 1):

        #update the similarity matrix
        S += local_random_walk_link_prediction(A, u)

    #removing the diagonal and nodes that are already connected
    return clean_link_prediction_matrix(S, A)

'''
A = sparse matrix representing a network
f = function that calculates similarity between nodes, it is of type f(A, *args)
args = additional arguments of function f as a python list
return = precision of the predicted links considering real links
'''
def precision(A, f, args = None):
    
    assert type(A) == sp.sparse.csr_matrix
    assert (args is None) or (type(args) == list) or (type(args) == tuple)

    N = A.shape[0]

    #findig out where there is already a link between nodes
    x,y = np.where(sp.sparse.triu(A).toarray() == 1)

    #finding out how many link choiches there are
    possible_choiches = np.arange(len(x))

    #getting the size of the probe set
    L = x.shape[0] // 10

    #getting one tenth of the overall links as probe set
    choiches = np.random.choice(possible_choiches, size=(L), replace=False)
    p = np.concatenate([x[choiches].reshape(-1, 1), y[choiches].reshape(-1, 1)], axis = 1).T
    values = np.ones((p.shape[1]))

    #building the probe matrix
    A_p = sp.sparse.csr_matrix((values, p), shape = (N, N))
    A_p = A_p + A_p.T

    #getting the test set matrix
    A_t = A - A_p

    #getting the similarity matrix from the test set
    if args is not None:
        S_t = f(A_t, *args)
    else:
        S_t = f(A_t)

    #getting L new predictions
    maximas = get_new_links(S_t, L)
    
    #computing the percentage of predicted links that are in the probe set
    counts = 0
    for amax in maximas:
        if A_p[amax[0], amax[1]] == 1:
            counts += 1

    return counts/L

'''
S = similarity matrix
n = number of links to be predicted
returns = the n most likely links
'''
def get_new_links(S, n):

    assert type(S) == np.ndarray
    assert (type(n) == int) and (n > 0) 
    maximas = []

    #repeat n times
    for _ in range(n):

        #get the nodes with the highest similarity
        amax = np.unravel_index(np.argmax(S), S.shape)

        #remove the similarity value so the same links aren't picked multiple times
        S[amax[0], amax[1]] = 0
        S[amax[1], amax[0]] = 0
        maximas.append(amax)

    return maximas

'''
A = sparse matrix representing the network
S = similarity matrix
return = similarity matrix without diagonal (self loops) and without links that already exist
'''
def clean_link_prediction_matrix(S, A):
    
    assert type(A) == sp.sparse.csr_matrix
    assert type(S) == np.ndarray

    #sets all the values in the main diagonal to 0
    np.fill_diagonal(S, 0)

    #sets all the values relative to existing links to 0
    S[A.toarray() == 1] = 0

    return S

'''
A = sparse matrix
division = array dividing the various communities
return = modularity
'''
def get_modularity(A, division):

    assert type(A) == sp.sparse.csr_matrix
    assert type(division) == np.ndarray

    d = get_degrees(A)
    D = np.sum(d)

    K = d@d.T
    c = division.reshape(-1, 1)
    
    return (np.sum(A[c == c.T]) - np.sum(K[c == c.T])/D)/D