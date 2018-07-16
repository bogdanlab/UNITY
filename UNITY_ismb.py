import scipy.misc as sm
import numpy as np
import math
import random
import scipy.stats as st
import sys
from optparse import OptionParser
from numpy import linalg as la

from auxilary import *

def nearPSD(A,epsilon=0):
   n = A.shape[0]
   eigval, eigvec = np.linalg.eig(A)
   val = np.matrix(np.maximum(eigval,epsilon))
   vec = np.matrix(eigvec)
   T = 1/(np.multiply(vec,vec) * val.T)
   T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
   B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
   out = B*B.T
   return(out)

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


# global variables for prior
lam1 = .2
lam2 = .2
lam3 = .2
lam4 = .2

def load_sumstats(file1, file2):
    z1 = np.loadtxt(file1)
    z2 = np.loadtxt(file2)
    print "sumstats loaded..."
    return z1, z2


def simulate(A00, A10, A01, A11, H1, H2, rho, M, N1, N2, Ns):
    rho_e = 0

    sig_11 = (H1 / (M * (A11 + A10)))
    sig_22 = (H2 / (M * (A11 + A01)))
    sig_12 = (math.sqrt(H1) * math.sqrt(H2) * rho / (M * A11))
    sig_21 = sig_12

    c = np.random.multinomial(1, [A00, A10, A01, A11], M)
    C1 = np.empty(M)
    C2 = np.empty(M)

    for m in range(0, M):
        if c[m, 0] == 1:
            C1[m] = 0
            C2[m] = 0
        elif c[m, 1] == 1:
            C1[m] = 1
            C2[m] = 0
        elif c[m, 2] == 1:
            C1[m] = 0
            C2[m] = 1
        else:
            C1[m] = 1
            C2[m] = 1

    mu = [0, 0]
    cov = [[sig_11, sig_12], [sig_21, sig_22]]

    gamma = np.random.multivariate_normal(mu, cov, M)

    beta1 = np.empty(M)
    beta2 = np.empty(M)
    for m in range(0, M):
        beta1[m] = gamma[m, 0] * (c[m, 1] + c[m, 3])
        beta2[m] = gamma[m, 1] * (c[m, 2] + c[m, 3])

    Sig_11 = (1 - H1) / N1
    Sig_22 = (1 - H2) / N2

    SIGMA_BETA1 = Sig_11
    SIGMA_BETA2 = Sig_22

    mu = np.concatenate([beta1, beta2])

    z1 = np.empty(M)
    z2 = np.empty(M)

    for m in range(0, M):
      z1[m] = np.random.normal(beta1[m],math.sqrt(SIGMA_BETA1), 1)
      z2[m] = np.random.normal(beta2[m], math.sqrt(SIGMA_BETA2), 1)

    print "Simulation done..."

    return z1, z2


def q_rand(a_old, sigma_gamma_old, sigma_beta1_old, sigma_beta2_old, func):

    if func == "better":
        # a param
        B = 10
        alpha1 =  lam1 + a_old[0]*B
        alpha2 =  lam2 + a_old[1]*B
        alpha3 =  lam3 + a_old[2]*B
        alpha4 =  lam4 + a_old[3]*B

        if alpha1 < .1:
            alpha1 = .1
        if alpha2 < .1:
            alpha2 = .1
        if alpha3 < .1:
            alpha3 = .1
        if alpha4 < .1:
            alpha4 = .1


        alpha_vec = [alpha1, alpha2, alpha3, alpha4]

        try:
            a_star = st.dirichlet.rvs(alpha_vec)
        except:
            print "error in drawing dirch"

        a_star = a_star.ravel()

        r = [a_star, sigma_gamma_old, sigma_beta1_old, sigma_beta2_old]

    else:
        r = 0

    return r, alpha_vec


def log_q_pdf(params_star, params_old, func, alpha_vec):

    a_star, sigma_gamma_star, sigma_beta1_star, sigma_beta2_star = params_star

    if func == "better":
        # a param

        try:
            d_a = st.dirichlet.pdf(x=a_star, alpha=alpha_vec)
        except:
            print "a-vector not summing to 1"

        if d_a == 0:
          d_a = 2.2250738585072014e-308
        try:
            log_q = math.log(d_a)
        except:
            log_q = 0
            print "log error in q_pdf"

    else:
        log_q = 0

    return log_q


def log_p_pdf_fast(z1, z2, params, H1, H2, rho, N1, N2, M):

    a, sigma_gamma, sigma_beta1, sigma_beta2 = params
    a00, a10, a01, a11 = a

    cov_e_coef = 0

    sig_gam1 = H1 / ((a11 + a10) * M)
    sig_gam2 = H2 / ((a11 + a01) * M)
    sig_gam3 = (math.sqrt(H1) * math.sqrt(H2) * rho) / (a11 * M)

    sigma_beta1 = (1-H1)/N1
    sigma_beta2 = (1-H2)/N2

    cov_e = (cov_e_coef * math.sqrt(sigma_beta1) * math.sqrt(sigma_beta2)) / float(N1 * N2)
    cov_00 = np.asarray([[sigma_beta1, cov_e], [cov_e, sigma_beta2]])
    cov_10 = np.asarray([[(sig_gam1 + sigma_beta1), cov_e], [cov_e, sigma_beta2]])
    cov_01 = np.asarray([[sigma_beta1, cov_e], [cov_e, (sig_gam2 + sigma_beta2)]])
    cov_11 = np.asarray([[sig_gam1 + sigma_beta1, sig_gam3 + cov_e], [sig_gam3 + cov_e, sig_gam2 + sigma_beta2 ]])
    mu = [0, 0]

    z = np.column_stack((z1, z2))

    pdf_00 = st.multivariate_normal.pdf(z, mean=mu, cov=cov_00, allow_singular=True)
    pdf_10 = st.multivariate_normal.pdf(z, mean=mu, cov=cov_10, allow_singular=True)
    pdf_01 = st.multivariate_normal.pdf(z, mean=mu, cov=cov_01, allow_singular=True)
    try:
        pdf_11 = st.multivariate_normal.pdf(z, mean=mu, cov=cov_11, allow_singular=True)
    except:
        cov_11_alt = nearPSD(cov_11)
        pdf_11 = st.multivariate_normal.pdf(z, mean=mu, cov=cov_11_alt, allow_singular=True)

    d_00 = np.multiply(pdf_00, a00)
    d_10 = np.multiply(pdf_10, a10)
    d_01 = np.multiply(pdf_01, a01)
    d_11 = np.multiply(pdf_11, a11)

    # find values in array that are zero
    zero_inds_00 = np.nonzero(d_00 == 0)
    zero_inds_10 = np.nonzero(d_10 == 0)
    zero_inds_01 = np.nonzero(d_01 == 0)
    zero_inds_11 = np.nonzero(d_11 == 0)

    # replace zero values
    d_00[zero_inds_00] = LOG_MIN
    d_10[zero_inds_10] = LOG_MIN
    d_01[zero_inds_01] = LOG_MIN
    d_11[zero_inds_11] = LOG_MIN

    # logsumexp for each SNP to get likelihood for each SNP
    snp_pdfs = logsumexp_vector([np.log(d_00), np.log(d_10), np.log(d_01), np.log(d_11)])

    # sum over every SNPs likelihood
    log_dbetahat = np.sum(snp_pdfs)


    d_a = st.dirichlet.pdf(a, [lam1, lam2, lam3, lam4])
    if d_a == 0:
        d_a = LOG_MIN

    d_H1 = variance_pdf(H1)
    if d_H1 == 0:
        d_H1 = LOG_MIN

    d_H2 = variance_pdf(H2)
    if d_H2 == 0:
        d_H2 = LOG_MIN

    d_rho = rho_pdf(rho)
    if d_rho == 0:
        d_rho = LOG_MIN

    log_p = log_dbetahat + math.log(d_a) + math.log(d_H1) + math.log(d_H2) + math.log(d_rho)

    return log_p



def log_p_pdf(z1, z2, params, H1, H2, rho, N1, N2, M):

    a, sigma_gamma, sigma_beta1, sigma_beta2 = params
    a00, a10, a01, a11 = a
    log_dbetahat = 0

    sig_gam1 = H1/((a11 + a10)*M)
    sig_gam2 = H2/((a11+a01)*M)
    sig_gam3 = (math.sqrt(H1)*math.sqrt(H2)*rho )/(a11*M)

    for m in range(0, M):
        z = [z1[m], z2[m]]

        placeholder = 2.2250738585072413e-308

        cov_00 = np.asarray([[sigma_beta1, 0], [0, sigma_beta2]])

        x00 = st.multivariate_normal.pdf(x=z, mean=[0, 0], cov=cov_00)
        if x00 == 0:
            x00 = 1.7976931348623157e-308
        x1 = x00*a00

        try:
          math.log(x1)
        except:
          x1 = placeholder

        cov_10 = np.asarray([[(sig_gam1 + sigma_beta1), 0], [0, sigma_beta2]])

        x10 = st.multivariate_normal.pdf(x=z, mean=[0, 0], cov=cov_10)
        if x10 == 0:
            x10 = 1.7976931348623157e-308
        x2 = x10 * a10

        try:
          math.log(x2)
        except:
          x2 = placeholder

        cov_01 = np.asarray([[sigma_beta1, 0],[0, (sig_gam2 + sigma_beta2)]])

        x01 = st.multivariate_normal.pdf(x=z, mean=[0, 0], cov=cov_01)
        if x01 == 0:
            x01 = 1.7976931348623157e-308
        x3 = x01*a01
        try:
          math.log(x3)
        except:
          x3 = placeholder

        cov_11 = np.asarray([[sig_gam1 + sigma_beta1, sig_gam3],[sig_gam3, sig_gam2 + sigma_beta2]])
        try:
            x11 = st.multivariate_normal.pdf(x=z, mean=[0,0 ], cov= cov_11)
        except:
#            print cov_11
            cov_11_alt = nearPSD(cov_11)
#            print cov_11_alt
            x11 = st.multivariate_normal.pdf(x=z, mean=[0,0 ], cov= cov_11_alt, allow_singular=True)
        if x11 == 0:
            x11 = 1.7976931348623157e-308
        x4 = x11*a11
        try:
          math.log(x11*a11)
        except:
          x4 = placeholder

        try:
          log_dbetahat += sm.logsumexp([math.log(x1), math.log(x2), math.log(x3), math.log(x4)])
        except:
          print "pdf error"

    # end for-loop

    temp = st.dirichlet.pdf(a, [lam1, lam2, lam3, lam4])
    log_da = math.log(temp)

    log_p = log_dbetahat + log_da

    return log_p


def accept_prob(z1, z2, params_star, params_old, func, alpha_vec_star, alpha_vec_old, H1, H2, rho, N1, N2, M):
    #a_star, sigma_gamma_star, sigma_beta_star = params_star
    #a_old, sigma_gamma_old, sigma_beta_old = params_old

    log_q_star = log_q_pdf(params_star, params_old, func, alpha_vec_star)

    log_q_old = log_q_pdf(params_old, params_star, func, alpha_vec_old)

    log_p_star = log_p_pdf_fast(z1, z2, params_star, H1, H2, rho,N1, N2, M)

    log_p_old = log_p_pdf_fast(z1, z2, params_old, H1, H2, rho,N1, N2, M)

    try:
        r = (log_p_star - log_p_old) + (log_q_old - log_q_star)
    except:
        print "log error"

    if r < 709:
        try:
            R = math.exp(r)
        except:
            print "exp error"
    else:
        R = 100

    accept = min(1, R)

    return accept


def run_experiment(H1, H2, rho, N1, N2, M, z1, z2, uid, ITS, f):
    # burnin
    BURN = ITS/4

    # calculating acceptance probabilities
    ACCEPT_a = 0

    # store estimates in lists
    a00_t = 0
    a10_t = 0
    a01_t = 0
    a11_t = 0

    a00_list = []
    a10_list = []
    a01_list = []
    a11_list = []

    func = "better"

    # initialize for time step 0
    a_old = st.dirichlet.rvs([lam1, lam2, lam3, lam4])
    a_old = a_old.ravel()
    a00, a10, a01, a11 = a_old
    sig_gam1 = H1 / ((a11 + a10) * M)
    sig_gam2 = H2 / ((a11 + a01) * M)
    sig_gam3 = (math.sqrt(H1) * math.sqrt(H2) * rho) / (a11 * M)

    sigma_gamma_old = np.asarray([[sig_gam1, sig_gam3],[sig_gam3, sig_gam2]])
    sigma_beta1_old = (1 - H1) / N1
    sigma_beta2_old = (1 - H2)/N2

    alpha_vec_old = [lam1, lam2, lam3, lam4]

    for i in range(0, ITS):

        [a_star, sigma_gamma_star, sigma_beta1_star, sigma_beta2_star], alpha_vec_star \
            = q_rand(a_old, sigma_gamma_old, sigma_beta1_old, sigma_beta2_old, func)

        params_old = [a_old, sigma_gamma_old, sigma_beta1_old, sigma_beta2_old]

        # accept a
        params_star = [a_star, sigma_gamma_old, sigma_beta1_old, sigma_beta2_old]

        accept_a = accept_prob(z1, z2, params_star, params_old, func, alpha_vec_star, alpha_vec_old, H1, H2, rho, N1, N2, M)

        u = st.uniform.rvs(size=1)
        if u < accept_a:
            a = a_star
            alpha_vec = alpha_vec_star
            ACCEPT_a += 1
        else:
            a = a_old
            alpha_vec = alpha_vec_old

        a_old = a
        alpha_vec_old = alpha_vec


        # debugging
        if i%50 == 0:
            print "Iteration: %d\n" % i
            print "p00: %.4f, p10: %.4f, p01: %.4f, p11: %.4f \n" % (a_old[0], a_old[1], a_old[2], a_old[3])

            f.write("Iteration: %d\n" % i)
            f.write("p00: %.4f, p10: %.4f, p01: %.4f, p11: %.4f \n" % (a_old[0], a_old[1], a_old[2], a_old[3]))

            sys.stdout.flush()

        # save the values
        if i >= BURN:
          a00_t += a_old[0]
          a10_t += a_old[1]
          a01_t += a_old[2]
          a11_t += a_old[3]

          a00_list.append(a_old[0])
          a10_list.append(a_old[1])
          a01_list.append(a_old[2])
          a11_list.append(a_old[3])

    a00_med = a00_t/float(ITS-BURN)
    a10_med = a10_t/float(ITS-BURN)
    a01_med = a01_t/float(ITS-BURN)
    a11_med = a11_t/float(ITS-BURN)

    a00_std = np.std(a00_list)
    a10_std = np.std(a10_list)
    a01_std = np.std(a01_list)
    a11_std = np.std(a11_list)

    print "mcmc-p00: %.6g" % a00_med
    print "mcmc-p10: %.6g" % a10_med
    print "mcmc-p01: %.6g" % a01_med
    print "mcmc-p11: %.6g" % a11_med

    print "standard deviations...\n"
    print "p00_std: %.6g" % a00_std
    print "p10_std: %.6g" % a10_std
    print "p01_std: %.6g" % a01_std
    print "p11_std: %.6g" % a11_std

    f.write("mcmc-p00: %.6g \n" % a00_med)
    f.write("mcmc-p10: %.6g \n" % a10_med)
    f.write("mcmc-p01: %.6g \n" % a01_med)
    f.write("mcmc-p11: %.6g \n" % a11_med)

    f.write("p00_std: %.6g\n" % a00_std)
    f.write("p10_std: %.6g\n" % a10_std)
    f.write("p01_std: %.6g\n" % a01_std)
    f.write("p11_std: %.6g\n" % a11_std)

    f.write('\n')
    f.flush()



def main():

    print "Starting..."

    # get input options
    parser = OptionParser()
    parser.add_option("--s", "--seed", dest="seed",default=1)
    parser.add_option("--H1", "--H1", dest="H1")
    parser.add_option("--H2", "--H2", dest="H2")
    parser.add_option("--rho", "--rho", dest="rho", default=0)
    parser.add_option("--M", "--M", dest="M", default=500)
    parser.add_option("--N1", "--N1", dest="N1", default=1000)
    parser.add_option("--N2", "--N2", dest="N2", default=1000)
    parser.add_option("--Ns", "--Ns", dest="Ns", default=0)
    parser.add_option("--A00", "--A00", dest="A00")
    parser.add_option("--A10", "--A10", dest="A10")
    parser.add_option("--A01", "--A01", dest="A01")
    parser.add_option("--A11", "--A11", dest="A11")
    parser.add_option("--id", "--id", dest="id", default="unique_id")
    parser.add_option("--ITS", "--ITS", dest="ITS", default=500)
    parser.add_option("--sim", "--sim", dest="sim", default="N")
    parser.add_option("--file1", "--file1", dest="file1")
    parser.add_option("--file2", "--file2", dest="file2")

    (options, args) = parser.parse_args()
    seed =int(options.seed)
    H1 = float(options.H1)
    H2 = float(options.H2)
    rho = float(options.rho)
    M = int(options.M)
    N1 = int(options.N1)
    N2 = int(options.N2)
    Ns = int(options.Ns)
    sim = options.sim
    if sim == "Y":
        A00 = float(options.A00)
        A10 = float(options.A10)
        A01 = float(options.A01)
        A11 = float(options.A11)
        if (A00 + A10 + A01 + A11)!= 1:
           print "p-vector does not equal 1...exiting\n"
           sys.exit(1)
    ID = options.id
    ITS = int(options.ITS)

    file1 = options.file1
    file2 = options.file2


    np.random.seed(seed)
    random.seed(seed)

    if sim == "Y":
        z1, z2 = simulate(A00, A10, A01, A11, H1, H2, rho, M, N1, N2, Ns)
    else:
        if file1 is None and file2 is  None:
            print "Error: Need sumstats file!"
            sys.exit(0)
        else:
            z1, z2 = load_sumstats(file1, file2)

    f = open("out.%s.%d" % (ID, seed), 'w')

    run_experiment(H1, H2, rho, N1, N2, M, z1, z2, ID, ITS, f)

    if sim == "Y":
        print "True params: A00:%.4f, A10:%.4f, A01:%.4f, A11:%.4f, H1:%.4f, H2:%.4f, rho:%.4f" % (A00, A10, A01, A11, H1, H2, rho)
        f.write("True params: A00:%.4f, A10:%.4f, A01:%.4f, A11:%.4f, H1:%.4f, H2:%.4f, rho:%.4f" % (A00, A10, A01, A11, H1, H2, rho))
    f.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
