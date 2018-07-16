from UNITY_particle import *
import sys
from scipy.optimize import minimize
from auxilary import *

def neg_log_p_pdf_fast_pvec(params, z1, z2, H1, H2, rho, M, N1, N2):

    # untransformed proportions
    a00, a10, a01, a11, cov_e_coef = params

    # transform to p
    p = convert_to_p(a00, a10, a01, a11)
    a00, a10, a01, a11 = p

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
        log_p = LOG_MIN
        return (-1)*log_p

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

    a=[a00, a10, a01, a11]

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

    return (-1)*log_p


def neg_log_p_pdf_fast_joint(params, z1, z2, M, N1, N2):

    a00, a10, a01, a11, H1, H2, rho, cov_e_coef = params

   # debugging!
    rho = 0

    p = convert_to_p(a00, a10, a01, a11)

    p00, p10, p01, p11 = p

    sigma_beta1 = (1 - H1) / float(N1)
    sigma_beta2 = (1 - H2) / float(N2)

    sig_gam1 = H1 / (M * (a11 + a10))
    sig_gam2 = H2 / (M * (a11 + a01))
    sig_gam3 = (math.sqrt(H1) * math.sqrt(H2) * rho) / (M * a11)

    cov_e = (cov_e_coef * math.sqrt(sigma_beta1) * math.sqrt(sigma_beta2))/float(N1*N2)
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
        print cov_11
        exit(1)
    d_00 = np.multiply(pdf_00, p00)
    d_10 = np.multiply(pdf_10, p10)
    d_01 = np.multiply(pdf_01, p01)
    d_11 = np.multiply(pdf_11, p11)

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


    d_a = st.dirichlet.pdf(p, [lam1, lam2, lam3, lam4])
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

    d_sigma_beta1 = variance_pdf(sigma_beta1)
    if d_sigma_beta1 == 0:
        d_sigma_beta1 = LOG_MIN

    d_sigma_beta2 = variance_pdf(sigma_beta2)
    if d_sigma_beta2 == 0:
        d_sigma_beta2 = LOG_MIN

    log_p = log_dbetahat + math.log(d_a) + math.log(d_H1) + math.log(d_H2) + math.log(d_rho) \
            + math.log(d_sigma_beta1) + math.log(d_sigma_beta2)

    return (-1)*log_p


def q_pvec_rand(a_old, H1, H2, rho, M, alpha_vec_old):

    # a param
    B = 10
    alpha1 = lam1 + a_old[0]*B
    alpha2 = lam2 + a_old[1]*B
    alpha3 = lam3 + a_old[2]*B
    alpha4 = lam4 + a_old[3]*B

    alpha_vec = [alpha1, alpha2, alpha3, alpha4]

    try:
        a_star = st.dirichlet.rvs(alpha_vec)
    except:
        print "error in drawing dirch"

    a_star = a_star.ravel() 
    p00, p10, p01, p11 = a_star 

    cov = [[H1/(M*(p11+p10)), (math.sqrt(H1)*math.sqrt(H2)*rho)/(M*p11)],
           [(math.sqrt(H1)*math.sqrt(H2)*rho)/(M*p11), H2/(M*(p11+p01))]]

    try: # test to see if pos-semi-def
        isPosDef = np.random.multivariate_normal([0,0], cov)
    except: # not pos-sem-def, reject 
        a_star = a_old 
        alpha_vec = alpha_vec_old 

    r = a_star 

    return r, alpha_vec


def log_q_pdf(params_star, params_old, alpha_vec):

    a_star, H1_star, H2_star, rho_star, cov_e_star = params_star
    a_old, H1_old, H2_old, rho_old, cov_e_old = params_old

    # a param
    d_a = st.dirichlet.pdf(x=a_star, alpha=alpha_vec)

    # H1 param
    d_H1 = q_variance_pdf(H1_star, H1_old)

    # H2 param
    d_H2 = q_variance_pdf(H2_star, H1_old)

    # rho param
    d_rho = q_rho_pdf(rho_star, rho_old)

    if d_a == 0:
        d_a = LOG_MIN

    if d_H1 == 0:
        d_H1 = LOG_MIN

    if d_H2 == 0:
        d_H2 = LOG_MIN

    if d_rho == 0:
        d_rho = LOG_MIN

    log_q = math.log(d_a) + math.log(d_H1) + math.log(d_H2) + math.log(d_rho)


    return log_q


def log_p_pdf_fast(z1, z2, params, M, N1, N2):

    a, H1, H2, rho, cov_e_coef = params
    a00, a10, a01, a11 = a

    sigma_beta1 = (1-H1)/N1
    sigma_beta2 = (1-H2)/N2

    sig_gam1 = H1 / (M * (a11 + a10))
    sig_gam2 = H2 / (M * (a11 + a01))
    sig_gam3 = (math.sqrt(H1) * math.sqrt(H2) * rho) / (M * a11)

    # if no SNPs are shared, off-diagonal term doesn't make sense to keep
    if math.fabs(math.sqrt(H1) * math.sqrt(H2) * rho) > (M*a11):
        sig_gam3 = 0

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
#        print "Cov not pos-semi def..."
        log_p = LOG_MIN
        return log_p

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


def log_likelihood(z1, z2, params, M, N1, N2):

    a, H1, H2, rho, cov_e_coef = params
    a00, a10, a01, a11 = a

    sigma_beta1 = (1-H1)/N1
    sigma_beta2 = (1-H2)/N2

    sig_gam1 = H1 / (M * (a11 + a10))
    sig_gam2 = H2 / (M * (a11 + a01))
    sig_gam3 = (math.sqrt(H1) * math.sqrt(H2) * rho) / (M * a11)

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
        log_p = LOG_MIN
        return log_p

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

    return log_dbetahat


def accept_prob(z1, z2, params_star, params_old, alpha_vec_star, alpha_vec_old, M, N1, N2, i):

    log_q_star = log_q_pdf(params_star, params_old, alpha_vec_star)

    log_q_old = log_q_pdf(params_old, params_star, alpha_vec_old)
    
    log_p_star = log_p_pdf_fast(z1, z2, params_star, M, N1, N2)

    log_p_old = log_p_pdf_fast(z1, z2, params_old, M, N1, N2)

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


def initial_estimates(N1, N2, M, z1, z2, optimize, H1=None, H2=None, rho=None, V=None):
    print H1
    print H2
    print rho 

    # hold candidate starting values
    candidates = []
    densities = []

    for it in range(0, OPTIMIZATION_ITS):
        p0 = st.dirichlet.rvs([lam1, lam2, lam3, lam4])
        p0 = p0.ravel()

        # if LD
        # (!) assuming no shared invididuals!
        if V is not None:
            N_max = 1 / float(max(N1, N2))
            cov_e_0 = 0
            x0 = [p0[0], p0[1], p0[2], p0[3], cov_e_0]
            result = minimize(neg_importance_like, x0, tol=1e-8, method=optimize,
                              args=(z1, z2, H1, H2, rho, 0, N1, N2, V), jac=False,
                              bounds=[(0.00001, .99), (0.00001, .99), (0.00001, .99), (0.00001, .99)])
            #result = minimize(neg_log_p_pdf_fast_pvec, x0, tol=1e-8, method=optimize,
            #                  args=(z1, z2, H1, H2, rho, M, N1, N2, V), jac=False,
            #                  bounds=[(0.00001, .99), (0.00001, .99), (0.00001, .99), (0.00001, .99), (-1*N_max, N_max)])
            a00_est, a10_est, a01_est, a11_est, coef_e_est = result.x
            H1_est = H1
            H2_est = H2
            rho_est = rho

        # user provided H1, H2, rho but no LD
        elif H1 is not None and H2 is not None and rho is not None:
            print "Optimizing only p parameter..." 
            N_max = 1 / float(max(N1, N2))
            cov_e_0 = 0
            x0 = [p0[0], p0[1], p0[2], p0[3], cov_e_0]

            result = minimize(neg_log_p_pdf_fast_pvec, x0, tol=1e-8, method=optimize,
                               args=(z1, z2, H1, H2, rho, M, N1, N2), jac=False,
                               bounds=[(0.00001, .99), (0.00001, .99), (0.00001, .99), (0.00001, .99), (-1*N_max, N_max)])

            a00_est, a10_est, a01_est, a11_est, coef_e_est = result.x
            H1_est = H1
            H2_est = H2
            rho_est = rho

        # joint estimation of H1, H2, p-vec but no LD
        else:
            h1_0 = variance_rvs()
            h2_0 = variance_rvs()
            rho_0 = 0 # don't use rho in optimization
            N_max =1/float(max(N1, N2))
            cov_e_0 = 0

            x0 = [p0[0], p0[1], p0[2], p0[3], h1_0, h2_0, rho_0, cov_e_0]
            

            result = minimize(neg_log_p_pdf_fast_joint, x0, tol=1e-8, method=optimize,
                                  args=(z1, z2, M, N1, N2), jac=False,
                                  bounds=[(0.00001, .99), (0.00001, .99), (0.00001, .99), (0.00001, .99), (0.00001, .99),
                                          (0.00001, .99),  (-1, 1), (-1*N_max, N_max)])

            # get results
            a00_est, a10_est, a01_est, a11_est, H1_est, H2_est, rho_est, coef_e_est = result.x

        # transform a-est to p-est
        p00_est, p10_est, p01_est, p11_est = convert_to_p(a00_est, a10_est, a01_est, a11_est)

        # calculate density with MAP estimates
        params = [[p00_est, p10_est, p01_est, p11_est], H1_est, H2_est, rho_est, coef_e_est]
        density = log_p_pdf_fast(z1, z2, params, N1, N2, M)

        print "Candidate starting values (p-vec): %.4g, %.4g, %.4g, %.4g" % (p00_est, p10_est, p01_est, p11_est)
        print "Candidate starting values (H1): %.4g" % H1_est
        print "Candidate starting values (H2): %.4g" % H2_est
        print "Candidate starting values (rho): %.4g" % rho_est
        print "Candidate starting values (coef_e): %.6g" % coef_e_est
        print "Desity at MAP: %.4f" % density
        print '\n'

        candidates.append(params)
        densities.append(density)

    # end for-loop through candidate values

    # pick values with greatest density
    max_index = np.argmax(densities)

    # return initialization points with best MAP
    [p00_est, p10_est, p01_est, p11_est], H1_est, H2_est, rho_est, coef_e_est \
        = candidates[max_index]

    return p00_est, p10_est, p01_est, p11_est, H1_est, H2_est, rho_est, coef_e_est


def run_MCMC(init_values, N1, N2, M, z1, z2, ITS, A00_true, A10_true, A01_true, A11_true, H1_true, H2_true, rho_true, f):

    # print file heading
    f.write("p00, p10, p01, p11, H1, H2, rho, like, density\n")

    # use initial values from Step 1
    a00_old, a10_old, a01_old, a11_old, H1_old, H2_old, rho_old, cov_e_coef = init_values

    # if user gave values for p-vec, H1, H2, rho, use those
    if A00_true is not None and A10_true is not None and A01_true is not None and A11_true is not None: 
        a00_old = A00_true 
        a10_old = A10_true 
        a01_old = A01_true 
        a11_old = A11_true 
    if H1_true is not None:
        H1_old = H1_true
    if H2_true is not None:
        H2_old = H2_true
    if rho_true is not None:
        rho_old = rho_true

    # burnin
    BURN = ITS/4

    # calculating acceptance probabilities
    ACCEPT_a = 0
    ACCEPT_H1 = 0
    ACCEPT_H2 = 0
    ACCEPT_rho = 0

    # hold estimates in lists
    a00_list = []
    a10_list = []
    a01_list = []
    a11_list = []
    H1_list = []
    H2_list = []
    rho_list = []

    # store estimates
    a00_t = 0
    a10_t = 0
    a01_t = 0
    a11_t = 0
    H1_t = 0
    H2_t = 0
    rho_t = 0

    # fixed initialization
    a_old = [a00_old, a10_old, a01_old, a11_old]
    alpha_vec_old = [lam1, lam2, lam3, lam4]

    for i in range(0, ITS):
        
        placeholder = 1 

        if A00_true == None and A10_true == None and A01_true == None and A11_true == None: 
            # accept a
            a_star, alpha_vec_star = q_pvec_rand(a_old, H1_old, H2_old, rho_old, M, alpha_vec_old)
            params_old = [a_old, H1_old, H2_old, rho_old, cov_e_coef]
            params_star = [a_star, H1_old, H2_old, 0, cov_e_coef]
            accept_a = accept_prob(z1, z2, params_star, params_old, alpha_vec_star, alpha_vec_old, M, N1, N2, placeholder)

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

        # accept H1
        if H1_true == None:
            params_old = [a_old, H1_old, H2_old, rho_old, cov_e_coef]
            H1_star = q_variance_rvs_H1(a_old, H1_old, H2_old, rho_old, M)
            params_star = [a_old, H1_star, H2_old, rho_old, cov_e_coef]
            
            # check for pos-sem-def for star params 
            flag = check_pos_def(a_old, H1_star, H2_old, rho_old, M)
            
            if flag == False:
                print "Error: H1_star not pos-semi-def" 
                exit(1)

            accept_H1 = accept_prob(z1, z2, params_star, params_old, alpha_vec_star, alpha_vec_old, M, N1, N2, placeholder)
            u = st.uniform.rvs(size=1)
            if u < accept_H1:
                H1 = H1_star
                ACCEPT_H1 += 1
            else:
                H1 = H1_old
            
            H1_old = H1

        # accept H2
        if H2_true == None:
            params_old = [a_old, H1_old, H2_old, rho_old, cov_e_coef]
            H2_star = q_variance_rvs_H2(a_old, H1_old, H2_old, rho_old, M) 
            params_star = [a_old, H1_old, H2_star, rho_old, cov_e_coef]
            accept_H2 = accept_prob(z1, z2, params_star, params_old, alpha_vec_star, alpha_vec_old, M, N1, N2, placeholder)
            u = st.uniform.rvs(size=1)
            if u < accept_H2:
                H2 = H2_star
                ACCEPT_H2 += 1
            else:
                H2 = H2_old
            H2_old = H2


        # accept rho
        if rho_true == None:
            params_old = [a_old, H1_old, H2_old, rho_old, cov_e_coef]
            # draw rho-proposal unique to current H1, H2, p-vec
            rho_star = q_rho_rvs(rho_old, a_old, H1_old, H2_old, M)
            params_star = [a_old, H1_old, H2_old, rho_star, cov_e_coef]
            accept_rho = accept_prob(z1, z2, params_star, params_old, alpha_vec_old, alpha_vec_old, M, N1, N2, i)
            u = st.uniform.rvs(size=1)
            if u < accept_rho:
                rho = rho_star
                ACCEPT_rho += 1
            else:
                rho = rho_old
            rho_old = rho


        params = [a_old, H1_old, H2_old, rho_old, cov_e_coef]

        MAP_t = log_p_pdf_fast(z1, z2, params, M, N1, N2)
        like_t = log_likelihood(z1, z2, params, M, N1, N2)

        # debugging
        if i % 10 == 0:
            print '\n'
            print "Iteration %d" % i
            print "like(%d): %.4f" % (i, like_t)
            print "MAP(%d): %.4f" % (i, MAP_t)
            print "p-vec(%d): %.4f, %.4f, %.4f, %.4f" % (i, a_old[0], a_old[1], a_old[2], a_old[3])
            print "H1(%d): %.4f" % (i, H1_old)
            print "H2(%d): %.4f" % (i, H2_old)
            print "rho(%d): %.4f" % (i, rho_old)
            sys.stdout.flush()

        # save the values
        if i >= BURN:
            a00_t += a_old[0]
            a10_t += a_old[1]
            a01_t += a_old[2]
            a11_t += a_old[3]
            H1_t += H1_old
            H2_t += H2_old
            rho_t += rho_old

            a00_list.append(a_old[0])
            a10_list.append(a_old[1])
            a01_list.append(a_old[2])
            a11_list.append(a_old[3])
            H1_list.append(H1_old)
            H2_list.append(H2_old)
            rho_list.append(rho_old)

            # print to chain file
            f.write("%.6g, %.6g, %.6g, %.6g, %.4g, %.4g, %.4g, %.6f, %.6f\n" %
                    (a_old[0], a_old[1], a_old[2], a_old[3], H1_old, H2_old, rho_old, like_t, MAP_t))

    a00_med = a00_t/float(ITS-BURN)
    a10_med = a10_t/float(ITS-BURN)
    a01_med = a01_t/float(ITS-BURN)
    a11_med = a11_t/float(ITS-BURN)
    H1_med = H1_t/float(ITS-BURN)
    H2_med = H2_t/float(ITS-BURN)
    rho_med = rho_t/float(ITS-BURN)

    a00_std = np.std(a00_list)
    a10_std = np.std(a10_list)
    a01_std = np.std(a01_list)
    a11_std = np.std(a11_list)
    H1_std = np.std(H1_list)
    H2_std = np.std(H2_list)
    rho_std = np.std(rho_list)

    # rho percentiles 
    rho_first_quantile = np.percentile(rho_list, 2.5)
    rho_third_quantile = np.percentile(rho_list, 97.5)

    return a00_med, a10_med, a01_med, a11_med, H1_med, H2_med, rho_med, a00_std, a10_std, \
           a01_std, a11_std, H1_std, H2_std, rho_std, rho_first_quantile, rho_third_quantile 

