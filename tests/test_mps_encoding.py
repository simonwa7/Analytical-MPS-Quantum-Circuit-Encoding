from src.encoding.mps_encoding import (
    is_unitary,
    get_unitary_form_of_mps_site,
    encode_bond_dimension_two_mps_as_quantum_circuit,
    get_kak_decomposition_parameters_for_unitary,
)
import pytest
import numpy as np
import copy

SEED = 1234
np.random.seed(SEED)
