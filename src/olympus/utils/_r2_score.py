"""Replacement r2_score function for when sklearn is not available."""
import numpy as np

# ===============================================================================
# BSD 3-Clause License

# Copyright (c) 2007-2021 The scikit-learn developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def r2_score(y_true, y_pred):
    """:math:`R^2` (coefficient of determination) regression score function.
    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a :math:`R^2` score of 0.0.

    This function is taken from sklearn, and is used when sklearn is not available.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    z : float or ndarray of floats
        The :math:`R^2` score or ndarray of scores if 'multioutput' is
        'raw_values'.

    Notes
    -----
    This is not a symmetric function.
    Unlike most other scores, :math:`R^2` score may be negative (it need not
    actually be the square of a quantity R).
    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.
    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    """

    if len(y_pred) < 2:
        # TODO use proper logging
        print("R^2 score is not well-defined with less than two samples.")
        return np.nan

    if len(y_true.shape) == 1:
        y_true = y_true.reshape((len(y_true), 1))  # unsqueeze
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape((len(y_true), 1))  # unsqueeze

    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(
        axis=0, dtype=np.float64
    )
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (
        numerator[valid_score] / denominator[valid_score]
    )
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    return np.average(output_scores)
