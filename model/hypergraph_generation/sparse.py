import numpy as np
from sklearn.metrics import pairwise_distances
import scipy.sparse as sparse
import cvxpy as cp
from cvxpy.error import SolverError

from model.hypergraph_generation.hypergraph import HyperG



def gen_l1_hg(X, gamma, n_neighbors, log=False, with_feature=False):

    assert n_neighbors >= 1.
    assert X.ndim == 2

    n_nodes = X.shape[0]
    n_edges = n_nodes

    m_dist = pairwise_distances(X)
    m_neighbors = np.argsort(m_dist)[:, 0:n_neighbors+1]

    edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors+1)).reshape(-1)
    node_idx = []
    values = []

    for i_edge in range(n_edges):
        if log:
            pass
        neighbors = m_neighbors[i_edge].tolist()
        if i_edge in neighbors:
            neighbors.remove(i_edge)
        else:
            neighbors = neighbors[:-1]

        P = X[neighbors, :]
        v = X[i_edge, :]

        x = cp.Variable(P.shape[0], nonneg=True)
        objective = cp.Minimize(cp.norm((P.T@x).T-v, 2) + gamma * cp.norm(x, 1))
        prob = cp.Problem(objective)
        try:
            prob.solve()
        except SolverError:
            prob.solve(solver='SCS', verbose=False)

        node_idx.extend([i_edge] + neighbors)
        values.extend([1.] + x.value.tolist())

    node_idx = np.array(node_idx)
    values = np.array(values)

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))

    if with_feature:
        return HyperG(H, X=X)

    return HyperG(H)
