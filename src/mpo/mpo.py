import numpy as np

# Define the Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

def hamiltonian_to_mpo(hamiltonian, m):
    """
    Convert a Hamiltonian represented as a sum of tensor products of Pauli matrices
    into a matrix product operator (MPO) representation.
    
    Parameters:
        hamiltonian (dict): A dictionary representing the Hamiltonian as a sum of tensor products of Pauli matrices.
                            The keys are tuples of length m representing the tensor products, and the values are the
                            corresponding coefficients. For example, {(0, 1): 0.5, (1, 2): -0.3} represents the Hamiltonian
                            0.5 * sigma_x(0) * sigma_y(1) - 0.3 * sigma_y(1) * sigma_z(2).
        m (int): The number of sites in the system.
        
    Returns:
        mpo (list): A list of m matrices representing the MPO. Each matrix is of shape (D, D, 2, 2), where D is the bond
                    dimension of the MPO.
    """
    # Initialize the MPO with identity matrices
    mpo = [np.eye(2).reshape((1, 2, 2))] * m
    
    # Loop over the terms in the Hamiltonian and update the MPO
    for term, coefficient in hamiltonian.items():
        tensor = np.eye(1)
        for site in range(m):
            if site in term:
                tensor = np.kron(tensor, eval(f"sigma_{term.index(site)}"))
            else:
                tensor = np.kron(tensor, np.eye(2))
        tensor = coefficient * tensor.reshape((2,) * (2 * m))
        for site in range(m):
            if site in term:
                index = term.index(site)
                if index == 0:
                    mpo[site] = np.concatenate((mpo[site], tensor), axis=0)
                elif index == m - 1:
                    mpo[site] = np.concatenate((mpo[site], tensor), axis=1)
                else:
                    left = tensor.transpose((0, 2) + tuple(range(2 * index - 1)))
                    left = left.reshape((2 ** (2 * index), 2 ** (2 * (m - index - 1))))
                    right = tensor.transpose((2 * index + 1,) + tuple(range(2 * index)))
                    right = right.reshape((2 ** (2 * (m - index - 1)), 2 ** (2 * index)))
                    mpo[site] = np.concatenate((mpo[site][:, :, :, :2], left[:, None, :, :]), axis=1)
                    mpo[site] = np.concatenate((mpo[site], right[None, :, :, :]), axis=1)
            else:
                mpo[site] = np.concatenate((mpo[site], np.eye(2).reshape((1, 2, 2))), axis=1)
    
    # Add the final identity matrix to the last site of the MPO
    mpo[-1] = np.concatenate((mpo[-1], np.eye(2).reshape

