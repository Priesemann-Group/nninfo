"""Implementation of Reing et al. (2021) directed local differences between cohesion measures.

"""

import numpy as np
import scipy.stats as stats
from math import comb
from itertools import chain, combinations

import sxpid

def _get_entropy(pdf, var_id_list):
    """Calculate the entropy of a variable.

    Parameters
    ----------
    pdf_dict : dict
        Dictionary with keys being realization tuples and values their probability
    var_id_list : list
        List of variable IDs.

    Returns
    -------
    float
        Entropy of the variable.

    """
    delete_id_list = [i for i in range(pdf.nVar) if i not in var_id_list]
    marginalized_pdf = pdf.marginalize(*delete_id_list)
    return stats.entropy(marginalized_pdf.probs, base=2)

def powerset(iterable):
    """Return the powerset of an iterable.

    Parameters
    ----------
    iterable : iterable
        Iterable to calculate the powerset of.

    Returns
    -------
    list
        List of all subsets of the iterable.

    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def compute_C_k(pdf_dict, k):
    """This definition is sligntly different than in the paper
        compute_C_k(k) is equivalent to C_k(k+1) in the paper (Equation 9)
        But the paper is wrong, because it uses C_k(k) in equation 10
    """
    n = len(next(iter(pdf_dict.keys()))) - 1

    # Create SxPID Pdf
    pdf = sxpid.SxPID.PDF.from_dict(pdf_dict)


    # Find all combinations of length k of the variables
    source_combinations = list(combinations(range(n), k))

    target_index = n

    C_k = 1/comb(n, k) * sum([_get_entropy(pdf, source_combination + (target_index,))
                            - _get_entropy(pdf, source_combination) for source_combination in source_combinations])

    return C_k

def compute_C_k_diffs(pdf_dict):
    n = len(next(iter(pdf_dict.keys()))) - 1
    C_k = [compute_C_k(pdf_dict, k) for k in range(0, n + 1)]
    C_k_diffs = {f'C({k-1}||{k})':C_k[k-1] - C_k[k] for k in range(1, n + 1)}

    return C_k_diffs