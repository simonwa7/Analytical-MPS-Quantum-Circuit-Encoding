import numpy as np
from ncon import ncon
from src.mps.mps import get_truncated_mps


def _get_mps(wavefunction):
    number_of_sites = int(np.log2(len(wavefunction)))
    mps = [None] * number_of_sites

    if number_of_sites == 2:
        u, s, v = np.linalg.svd(wavefunction.reshape(2, 2), full_matrices=False)
        mps = [
            u.reshape(1, 2, 2),
            ncon([np.diag(s), v], [[-1, 1], [1, -2]]).reshape(2, 2, 1),
        ]
        return get_truncated_mps(mps, 2 ** number_of_sites)

    if number_of_sites == 3:
        u, s, v = np.linalg.svd(wavefunction.reshape(2, 4), full_matrices=False)
        mps[0] = u.reshape(1, 2, 2)
        wavefunction = ncon([np.diag(s), v], [[-1, 1], [1, -2]])
        u, s, v = np.linalg.svd(wavefunction.reshape(4, 2), full_matrices=False)
        mps[1] = u.reshape(2, 2, 2)
        mps[2] = ncon([np.diag(s), v], [[-1, 1], [1, -2]]).reshape(2, 2, 1)
        return get_truncated_mps(mps, 2 ** number_of_sites)

    if number_of_sites == 4:
        u, s, v = np.linalg.svd(wavefunction.reshape(2, 8), full_matrices=False)
        mps[0] = u.reshape(1, 2, 2)
        wavefunction = ncon([np.diag(s), v], [[-1, 1], [1, -2]])
        u, s, v = np.linalg.svd(wavefunction.reshape(4, 4), full_matrices=False)
        mps[1] = u.reshape(2, 2, 4)
        wavefunction = ncon([np.diag(s), v], [[-1, 1], [1, -2]])
        u, s, v = np.linalg.svd(wavefunction.reshape(8, 2), full_matrices=False)
        mps[2] = u.reshape(4, 2, 2)
        mps[3] = ncon([np.diag(s), v], [[-1, 1], [1, -2]]).reshape(2, 2, 1)
        return get_truncated_mps(mps, 2 ** number_of_sites)

    if number_of_sites == 5:
        u, s, v = np.linalg.svd(wavefunction.reshape(2, 16), full_matrices=False)
        mps[0] = u.reshape(1, 2, 2)
        wavefunction = ncon([np.diag(s), v], [[-1, 1], [1, -2]])
        u, s, v = np.linalg.svd(wavefunction.reshape(4, 8), full_matrices=False)
        mps[1] = u.reshape(2, 2, 4)
        wavefunction = ncon([np.diag(s), v], [[-1, 1], [1, -2]])
        u, s, v = np.linalg.svd(wavefunction.reshape(8, 4), full_matrices=False)
        mps[2] = u.reshape(4, 2, 4)
        wavefunction = ncon([np.diag(s), v], [[-1, 1], [1, -2]])
        u, s, v = np.linalg.svd(wavefunction.reshape(8, 2), full_matrices=False)
        mps[3] = u.reshape(4, 2, 2)
        mps[4] = ncon([np.diag(s), v], [[-1, 1], [1, -2]]).reshape(2, 2, 1)
        return get_truncated_mps(mps, 2 ** number_of_sites)

    if number_of_sites == 6:
        u, s, v = np.linalg.svd(wavefunction.reshape(2, 32), full_matrices=False)
        mps[0] = u.reshape(1, 2, 2)
        wavefunction = ncon([np.diag(s), v], [[-1, 1], [1, -2]])
        u, s, v = np.linalg.svd(wavefunction.reshape(4, 16), full_matrices=False)
        mps[1] = u.reshape(2, 2, 4)
        wavefunction = ncon([np.diag(s), v], [[-1, 1], [1, -2]])
        u, s, v = np.linalg.svd(wavefunction.reshape(8, 8), full_matrices=False)
        mps[2] = u.reshape(4, 2, 8)
        wavefunction = ncon([np.diag(s), v], [[-1, 1], [1, -2]])
        u, s, v = np.linalg.svd(wavefunction.reshape(16, 4), full_matrices=False)
        mps[3] = u.reshape(8, 2, 4)
        wavefunction = ncon([np.diag(s), v], [[-1, 1], [1, -2]])
        u, s, v = np.linalg.svd(wavefunction.reshape(8, 2), full_matrices=False)
        mps[4] = u.reshape(4, 2, 2)
        mps[5] = ncon([np.diag(s), v], [[-1, 1], [1, -2]]).reshape(2, 2, 1)
        return get_truncated_mps(mps, 2 ** number_of_sites)
