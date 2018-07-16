import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as st
from scipy.misc import logsumexp
from scipy.special import logit

# iteration params
#PARTICLES = 5000
OPTIMIZATION_ITS = 10

# global variables for prior
lam1 = .2
lam2 = .2
lam3 = .2
lam4 = .2

invgamma_prior = 1
invgamma_prior_scale = .5

LOG_MAX = 1.7976931348623157e+308
LOG_MIN = 1.7976931348623157e-308
EXP_MAX = 709


def print_header(N1, N2, Ns_sim, Ns, M, ITS, A00, A10, A01, A11, A00_sim, A10_sim, A01_sim, A11_sim, H1, H2, H1_sim, H2_sim, rho, rho_sim, true_gcor,
                 rho_e_sim, sim, file1, file2, uid, f, LD_file=None):


    print "Name: %s" % uid
    f.write("Name: %s \n" % uid)

    if Ns is not None: # user knows sampler overlap
        print "Experiment: N1: %d, N2: %d, Ns: %d, M: %d, ITS: %d" % (N1, N2, Ns, M, ITS)
        f.write("Experiment: N1: %d, N2: %d, Ns: %d, M: %d, ITS: %d\n" % (N1, N2, Ns, M, ITS))
    else: # user does not know sample overlap
        print "Experiment: N1: %d, N2: %d, M: %d, ITS: %d" % (N1, N2, M, ITS)
        f.write("Experiment: N1: %d, N2: %d, M: %d, ITS: %d \n" % (N1, N2, M, ITS))
        print "Unknown number of shared individuals...will infer with MAP"
        f.write("Unknown number of shared individuals...will infer with MAP\n")

    if sim == "Y":
        print "Simulating with: A00: %.4f, A10: %.4f, A01: %.4f, A11: %.4f, H1: %.4f, H2: %.4f, " \
              "rho: %.4f, rho-true: %.4f, rho_e: %.4f, Ns: %d" % (
            A00_sim, A10_sim, A01_sim, A11_sim, H1_sim, H2_sim, rho_sim, true_gcor, rho_e_sim, Ns_sim)

        f.write(
            "Simulating with: A00: %.4f, A10: %.4f, A01: %.4f, A11: %.4f, H1: %.4f, H2: %.4f, "
            "rho: %.4f, rho-true: %.4f, rho_e: %.4f, Ns: %.d \n" % (
                A00_sim, A10_sim, A01_sim, A11_sim, H1_sim, H2_sim, rho_sim, true_gcor, rho_e_sim, Ns_sim))

        if LD_file is not None:
            print "Simulating with LD...user provided file: %s" % LD_file
            f.write("Simulating with LD...user provided file: %s\n" % LD_file)

        if A00 is not None and A10 is not None and A01 is not None and A11 is not None:
            print "...user provided A00: %.4f, A10: %.4f, A01:%.4f, A11: %.4f" % (A00, A10, A01, A11)
            f.write("...user provided A00: %.4f, A10: %.4f, A01:%.4f, A11: %.4f \n" % (A00, A10, A01, A11))
        else:
            print("...going to infer A00, A10, A01, A11")
            f.write("...going to infer A00, A10, A01, A11\n")

        if H1 == None:
            print "...going to infer H1"
            f.write("...going to infer H1\n")
        else:
            print "...user provided H1 = %.4f" % (H1)
            f.write("...user provided H1 = %.4f\n" % (H1))

        if H2 == None:
            print "...going to infer H2"
            f.write("...going to infer H2\n")
        else:
            print "...user provide H2 = %.4f" % (H2)
            f.write("...user provide H2 = %.4f\n" % (H2))

        if rho == None:
            print "...going to infer rho"
            f.write("...going to infer rho\n")
        else:
            if true_gcor is None:
                print "...user provided rho = %.4f" % (rho)
                f.write("...user provided rho = %.4f\n" % (rho))
            else:
                print "...user provided rho = %.4f" % (rho)
                f.write("...user provided rho = %.4f\n" % (rho))

    else:  # not simulating
        print "Using files: %s, %s" % (file1, file2)
        f.write("Using files: %s, %s" % (file1, file2))

        if H1 == None:
            print "...going to infer H1"
            f.write("...going to infer H1\n")
        else:
            print "...user provided H1 = %.4f" % (H1)
            f.write("...user provided H1 = %.4f\n" % (H1))

        if H2 == None:
            print "...going to infer H2"
            f.write("...going to infer H2")
        else:
            print "...user provide H2 = %.4f" % (H2)
            f.write("...user provide H2 = %.4f\n" % (H2))

        if rho == None:
            print "...going to infer rho"
            f.write("...going to infer rho\n")
        else:
            if true_gcor is None:
                print "...user provided rho = %.4f" % (rho)
                f.write("...user provided rho = %.4f\n" % (rho))
            else:
                print "...using true gcorr = %.4f" % true_gcor
                f.write("...using true gcorr = %.4f\n" % true_gcor)


def simulate(A00, A10, A01, A11, H1, H2, rho, rho_e_sim, M, N1, N2, Ns, V=None):

    sig_11 = H1 / (M*(A11 + A10))
    sig_22 = H2 / (M*(A11 + A01))
    sig_12 = (math.sqrt(H1) * math.sqrt(H2) * rho )/ (M*(A11))
    sig_21 = sig_12

    c = np.random.multinomial(1, [A00, A10, A01, A11], M)
    counts = np.sum(c, axis=0)
#    print "True proportions: %.2f, %.2f, %.2f, %.2f" % \
#          (counts[0]/float(M), counts[1]/float(M), counts[2]/float(M), counts[3]/float(M))
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

    # debugging C-vector
    #np.savetxt("C1.txt", C1)
    #np.savetxt("C2.txt", C2)


    mu = [0, 0]
    cov = [[sig_11, sig_12], [sig_21, sig_22]]

    gamma = np.random.multivariate_normal(mu, cov, M)

    beta1 = np.empty(M)
    beta2 = np.empty(M)
    for m in range(0, M):
        beta1[m] = gamma[m, 0] * (c[m, 1] + c[m, 3])
        beta2[m] = gamma[m, 1] * (c[m, 2] + c[m, 3])

    true_corr_matrix = np.corrcoef(beta1, beta2)
    true_corr = true_corr_matrix[0,1]

    Sig_11 = (1 - H1) / N1
    Sig_22 = (1 - H2) / N2

    SIGMA_BETA1 = Sig_11
    SIGMA_BETA2 = Sig_22
    cov_e = rho_e_sim*math.sqrt(SIGMA_BETA1)*math.sqrt(SIGMA_BETA2)
    SIGMA_BETA3 = (cov_e*Ns)/float(N1*N2)
    SIGMA_BETA_cov = [[SIGMA_BETA1, SIGMA_BETA3],[SIGMA_BETA3, SIGMA_BETA2]]

    # (!) right now, assumes no shared individuals for case with LD!
    if V is not None: # simulate with LD
        mu1 = np.matmul(V, beta1)
        mu2 = np.matmul(V, beta2)
        cov1 = SIGMA_BETA1 * V
        cov2 = SIGMA_BETA2 * V
        z1 = st.multivariate_normal.rvs(mean=mu1, cov=cov1)
        z2 = st.multivariate_normal.rvs(mean=mu2, cov=cov2)
    else:  # no LD
        z1 = np.empty(M)
        z2 = np.empty(M)
        for m in range(0, M):
            mu = [beta1[m], beta2[m]]
            z1_m, z2_m = st.multivariate_normal.rvs(mu, SIGMA_BETA_cov)
            z1[m] = z1_m
            z2[m] = z2_m


    return z1, z2, true_corr


def trace_plot(param_list, ITS):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(range(0, ITS), param_list)

    fig.savefig('/Users/ruthiejohnson/Downloads/trace.png')


def isPosDef(cov):
    M = len(cov)
    flag = True
    try:
        temp = np.random.multivariate_normal(np.zeros(M), cov)
    except:
        flag = False
    return flag


def truncate_matrix(V):
    # make V pos-semi-def
    d, Q = np.linalg.eigh(V, UPLO='U')

    # reorder eigenvectors from inc to dec
    idx = d.argsort()[::-1]
    Q[:] = Q[:, idx]

    # truncate small eigenvalues for stability
    d_trun = truncate_eigenvalues(d)

    # mult decomp back together to get final V_trunc
    M1 = np.matmul(Q, np.diag(d_trun))
    V_trun = np.matmul(M1, np.matrix.transpose(Q))

    return V_trun


def variance_rvs():
    rv = st.beta.rvs(a=1, b=2)
    return rv


def variance_pdf(x):
    pdf = st.beta.pdf(x=x, a=1, b=2)
    return pdf


def rho_rvs():
    rv = st.norm.rvs(loc=0, scale=.50)
    return rv


def rho_pdf(x):
    pdf = st.norm.pdf(x=x, loc=0, scale=.50)
    return pdf

def check_pos_def(a, H1, H2, rho, M):
    flag = True 
    p00, p10, p01, p11 = a
    cov = [[H1/(M*(p11+p10)), (math.sqrt(H1)*math.sqrt(H2)*rho)/(M*p11)],
           [(math.sqrt(H1)*math.sqrt(H2)*rho)/(M*p11), H2/(M*(p11+p01))]]
    try:
        isPosDef = np.random.multivariate_normal([0,0], cov)
    except: # not pos-sem-def, reject
        flag = False 
    return flag 


def q_variance_rvs_H1(a_old, H1_old, H2_old, rho_old, M):
    # draw new value 
    H1 = sigmoid(st.norm.rvs(loc=logit(H1_old), scale=1.0))
    p00, p10, p01, p11 = a_old

    cov = [[H1/(M*(p11+p10)), (math.sqrt(H1)*math.sqrt(H2_old)*rho_old)/(M*p11)],
           [(math.sqrt(H1)*math.sqrt(H2_old)*rho_old)/(M*p11), H2_old/(M*(p11+p01))]]

    # check if pos-semi-def
    try:
        isPosDef = np.random.multivariate_normal([0,0], cov)
    except: # not pos-sem-def, reject 
        H1 = H1_old 

    return H1


def q_variance_rvs_H2(a_old, H1_old, H2_old, rho_old, M):
    # draw new value
    H2 = sigmoid(st.norm.rvs(loc=logit(H2_old), scale=1.0))
    p00, p10, p01, p11 = a_old

    cov = [[H1_old/(M*(p11+p10)), (math.sqrt(H1_old)*math.sqrt(H2)*rho_old)/(M*p11)],
           [(math.sqrt(H1_old)*math.sqrt(H2)*rho_old)/(M*p11), H2/(M*(p11+p01))]]

    # check if pos-semi-def
    try:
        isPosDef = np.random.multivariate_normal([0,0], cov)
    except: # not pos-sem-def, reject
        H2 = H2_old

    return H2


def q_variance_pdf(x, H_old):
    pdf = st.norm.pdf(x=logit(x), loc=logit(H_old), scale=1)
    return pdf


def q_rho_rvs(rho_old, p, H1, H2, M):
    p00, p10, p01, p11 = p
    rho = math.tanh(st.norm.rvs(loc=math.tan(rho_old), scale=.01))


    cov = [[H1/(M*(p11+p10)), (math.sqrt(H1)*math.sqrt(H2)*rho)/(M*p11)],
           [(math.sqrt(H1)*math.sqrt(H2)*rho)/(M*p11), H2/(M*(p11+p01))]]

    try:
        isPosDef = np.random.multivariate_normal([0,0], cov)
    except:
        rho = rho_old

    return rho


def q_rho_pdf(x, rho_old):

    pdf = st.norm.pdf(x=math.tan(x), loc=math.tan(rho_old), scale=.01)

    return pdf


def sigmoid_vec(x):
    y = np.multiply(-1, x)
    return np.divide(1, np.add(1, np.exp(y)))


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def logsumexp_vector(a, axis=0):
    if axis is None:
        return logsumexp(a)
    a = np.asarray(a)
    shp = list(a.shape)
    shp[axis] = 1
    a_max = a.max(axis=axis)
    s = np.log(np.exp(a - a_max.reshape(shp)).sum(axis=axis))
    lse  = a_max + s
    return lse


def load_sumstats(file1, file2):
    z1 = np.loadtxt(file1)
    z2 = np.loadtxt(file2)
    print "sumstats loaded..."
    return z1, z2


def convert_to_p(a00, a10, a01, a11):
    # ensures p-vec sums to 1

    if a00 > EXP_MAX:
        a00 = EXP_MAX
    if a10 > EXP_MAX:
        a10 = EXP_MAX
    if a01 > EXP_MAX:
        a01 = EXP_MAX
    if a11 > EXP_MAX:
        a11 = EXP_MAX

    C = a00 + a10 + a01 + a11

    p00 = a00 / C
    p10 = a10 / C
    p01 = a01 / C
    p11 = a11 / C

    p = [p00, p10, p01, p11]

    return p


def convert_to_H(G):
    H = sigmoid(G)
    return H


def convert_to_rho(pi):
    rho = math.tanh(pi)
    return rho


def propose_C(C_old, M):
    mu = np.sum(C_old)/float(M)
    #p = sigmoid(st.norm.rvs(logit(mu), 1))
    mu = .20
    #p = st.norm.rvs(mu, .025)

    p = mu
    if p > 1:
        p = mu
    elif p < 0:
        p = mu
    #print p
    C = np.zeros(M)


    for i in range(0, M):
        C[i] = st.bernoulli.rvs(p)
    #print "percent causal: %.2f" % (np.sum(C)/float(M))
    return C, p


def evaluate_C(C_star, p):
    mu = np.sum(C_star)/float(len(C_star))
    #pdf = st.norm.pdf(logit(mu), logit(p), 1)
    pdf = st.norm.pdf(mu, p, .025)

    return pdf


def propose_C1_C2_dir(C1_old, C2_old, M):

    p_old = calc_prop(C1_old, C2_old)

    B = 10
    alpha1 = lam1 + p_old[0] * B
    alpha2 = lam2 + p_old[1] * B
    alpha3 = lam3 + p_old[2] * B
    alpha4 = lam4 + p_old[3] * B

    alpha_vec = [alpha1, alpha2, alpha3, alpha4]

    p_new = st.dirichlet.rvs(alpha_vec)
    p00_new, p10_new, p01_new, p11_new = p_new.ravel()

    # draw new C1, C2
    c = st.multinomial.rvs(1, [p00_new, p10_new, p01_new, p11_new], M)

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

    p_new = calc_prop(C1, C2)
    print "Proposed C1, C1 proportions: %.4f, %.4f, %.4f, %.4f" % (p_new[0], p_new[1], p_new[2], p_new[3])
    return C1, C2, alpha_vec


def propose_C1_C2(C1_old, C2_old, M):
    ones = np.ones(M)
    C00 = np.multiply(np.subtract(ones, C1_old), np.subtract(ones, C2_old))
    C10 = np.multiply(C1_old, np.subtract(ones, C2_old))
    C01 = np.multiply(np.subtract(ones, C1_old), C2_old)
    C11 = np.multiply(C1_old, C2_old)

    A00 = np.sum(C00)/float(M)
    A10 = np.sum(C10)/float(M)
    A01 = np.sum(C01)/float(M)
    A11 = np.sum(C11)/float(M)

    p_old = calc_prop(C1_old, C2_old)

    # add noise to estimates
    A00_new = st.norm.rvs(A00, 1)
    A10_new = st.norm.rvs(A10, 1)
    A01_new = st.norm.rvs(A01, 1)
    A11_new = st.norm.rvs(A11, 1)

    p00_new, p10_new, p01_new, p11_new = convert_to_p(A00_new, A10_new, A01_new, A11_new)

    # draw new C1, C2
    c = st.multinomial.rvs(1, [p00_new, p10_new, p01_new, p11_new], M)

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

    p_new = calc_prop(C1, C2)
    print "Proposed C1, C1 proportions: %.4f, %.4f, %.4f, %.4f" % (p_new[0], p_new[1], p_new[2], p_new[3])
    return C1, C2, p_old


def log_evaluate_C1_C2_dir(C1, C2, p_old):
    p_star = calc_prop(C1, C2)
    d_a = st.dirichlet.pdf(x=p_star, alpha=p_old)
    if d_a == 0:
        d_a = LOG_MIN
    return math.log(d_a)


def log_evaluate_C1_C2(C1, C2, p_old):

    p_new  = calc_prop(C1, C2)
    p00_new, p10_new, p01_new, p11_new = p_new

    p00_old, p10_old, p01_old, p11_old = p_old

    d00 = st.norm.pdf(p00_new, p00_old, 1)
    d10 = st.norm.pdf(p10_new, p10_old, 1)
    d01 = st.norm.pdf(p01_new, p01_old, 1)
    d11 = st.norm.pdf(p11_new, p11_old, 1)

    if d00 == 0:
        d00 = LOG_MIN
    if d10 == 0:
        d10 = LOG_MIN
    if d01 == 0:
        d01 = LOG_MIN
    if d11 == 0:
        d11 = LOG_MIN

    log_dC = math.log(d00) + math.log(d10) + math.log(d01) + math.log(d11)

    return log_dC


def calc_prop(C1_old, C2_old):
    M = len(C1_old)
    ones = np.ones(M)
    C00 = np.multiply(np.subtract(ones, C1_old), np.subtract(ones, C2_old))
    C10 = np.multiply(C1_old, np.subtract(ones, C2_old))
    C01 = np.multiply(np.subtract(ones, C1_old), C2_old)
    C11 = np.multiply(C1_old, C2_old)

    C = np.column_stack((C00, C10, C01, C11))
    padding = 0
    if np.sum(C00) == 0:
        padding += 1
    if np.sum(C10) == 0:
        padding += 1
    if np.sum(C01) == 0:
        padding += 1
    if np.sum(C11) == 0:
        padding += 1

    A00 = (np.sum(C00) + padding )/ float(M+4*padding)
    A10 = (np.sum(C10) + padding )/ float(M+4*padding)
    A01 = (np.sum(C01) + padding )/ float(M+4*padding)
    A11 = (np.sum(C11) + padding )/ float(M+4*padding)

    return A00, A10, A01, A11


def truncate_eigenvalues(d):
    M = len(d)

    # order evaules in descending order
    d[::-1].sort()

    #running_sum = 0
    d_trun = np.zeros(M)

    # keep only positive evalues
    for i in range(0,M):
        if d[i] > 0:
            # keep evalue
            d_trun[i] = d[i]

    return d_trun
