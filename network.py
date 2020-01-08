import numpy as np
import scipy as sp
import pandas as pd
from random import uniform
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
