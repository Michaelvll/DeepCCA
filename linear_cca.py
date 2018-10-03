import numpy


def linear_cca(H1, H2, outdim_size):
    """
    An implementation of linear CCA
    # Arguments:
        H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
        outdim_size: specifies the number of new features
    # Returns
        A and B: the linear transformation matrices 
        mean1 and mean2: the means of data for both views
    """
    r1 = 1e-4
    r2 = 1e-4

    m = H1.shape[0]
    o = H1.shape[1]

    mean1 = numpy.mean(H1, axis=0)
    mean2 = numpy.mean(H2, axis=0)
    H1bar = H1 - numpy.tile(mean1, (m, 1))
    H2bar = H2 - numpy.tile(mean2, (m, 1))

    SigmaHat12 = (1.0 / (m - 1)) * numpy.dot(H1bar.T, H2bar)
    SigmaHat11 = (1.0 / (m - 1)) * numpy.dot(H1bar.T, H1bar) + r1 * numpy.identity(o)
    SigmaHat22 = (1.0 / (m - 1)) * numpy.dot(H2bar.T, H2bar) + r2 * numpy.identity(o)

    [D1, V1] = numpy.linalg.eigh(SigmaHat11)
    [D2, V2] = numpy.linalg.eigh(SigmaHat22)
    SigmaHat11RootInv = numpy.dot(numpy.dot(V1, numpy.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = numpy.dot(numpy.dot(V2, numpy.diag(D2 ** -0.5)), V2.T)

    Tval = numpy.dot(numpy.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    [U, D, V] = numpy.linalg.svd(Tval)
    V = V.T
    A = numpy.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
    B = numpy.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
    D = D[0:outdim_size]

    return A, B, mean1, mean2
