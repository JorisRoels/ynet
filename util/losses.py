import torch


def feature_regularization_loss(f_src, f_tar, method='coral', n_samples=None):
    """
    Compute the regularization loss between the feature representations (shape [B, C, Y, X]) of the two streams
    In case of high dimensionality, there is an option to subsample
    :param src: features of the source stream
    :param tar: features of the target stream
    :param method: regularization method ('coral' or 'mmd')
    :param optional n_samples: number of samples to be selected
    :return: regularization loss
    """

    # view features to [N, D] shape
    src = f_src.view(f_src.size(0), -1)
    tar = f_tar.view(f_tar.size(0), -1)

    if n_samples is None:
        fs = src
        ft = tar
    else:
        inds = np.random.choice(range(src.size(1)), n_samples, replace=False)
        fs = src[:, inds]
        ft = tar[:, inds]

    if method == 'coral':
        return coral(fs, ft)
    else:
        return mmd(fs, ft)


def coral(source, target):
    """
    Compute CORAL loss between two feature vectors (https://arxiv.org/abs/1607.01719)
    :param source: source vector [N_S, D]
    :param target: target vector [N_T, D]
    :return: CORAL loss
    """
    d = source.size(1)  # dim vector

    source_c = _compute_covariance(source)
    target_c = _compute_covariance(target)

    loss = torch.sum(torch.pow(source_c - target_c, 2)) / (4 * (d ** 2))

    return loss


def _compute_covariance(x):
    n = x.size(0)  # batch_size

    sum_column = torch.sum(x, dim=0, keepdim=True)
    term_mul_2 = torch.mm(sum_column.t(), sum_column) / n
    d_t_d = torch.mm(x.t(), x)

    return (d_t_d - term_mul_2) / (n - 1)


def mmd(source, target, gamma=10 ** 3):
    """
    Compute MMD loss between two feature vectors (https://arxiv.org/abs/1605.06636)
    :param source: source vector [N_S, D]
    :param target: target vector [N_T, D]
    :return: MMD loss
    """
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(source, target, [gamma])

    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=True)


def _mix_rbf_kernel(X, Y, sigma_list):
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    return mmd2
