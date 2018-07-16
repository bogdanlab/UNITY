from UNITY import *
import cProfile, pstats, StringIO
import random
from optparse import OptionParser

def main():

    # get input options
    parser = OptionParser()
    parser.add_option("--s", "--seed", dest="seed",default="7")
    parser.add_option("--H1", "--H1", dest="H1")
    parser.add_option("--H2", "--H2", dest="H2")
    parser.add_option("--rho", "--rho", dest="rho")
    parser.add_option("--H1_sim", "--H1_sim", dest="H1_sim")
    parser.add_option("--H2_sim", "--H2_sim", dest="H2_sim", default=0)
    parser.add_option("--rho_sim", "--rho_sim", dest="rho_sim")
    parser.add_option("--rho_e_sim", "--rho_e_sim", dest="rho_e_sim", default=0)
    parser.add_option("--M", "--M", dest="M", default=500)
    parser.add_option("--N1", "--N1", dest="N1", default=1000)
    parser.add_option("--N2", "--N2", dest="N2", default=1000)
    parser.add_option("--Ns", "--Ns", dest="Ns", default=None)
    parser.add_option("--Ns_sim", "--Ns_sim", dest="Ns_sim", default=0)
    parser.add_option("--A00_sim", "--A00_sim", dest="A00_sim", default=None)
    parser.add_option("--A10_sim", "--A10_sim", dest="A10_sim", default=None)
    parser.add_option("--A01_sim", "--A01_sim", dest="A01_sim", default=None)
    parser.add_option("--A11_sim", "--A11_sim", dest="A11_sim", default=None)
    parser.add_option("--A00", "--A00", dest="A00", default=None)
    parser.add_option("--A10", "--A10", dest="A10", default=None)
    parser.add_option("--A01", "--A01", dest="A01", default=None)
    parser.add_option("--A11", "--A11", dest="A11", default=None)
    parser.add_option("--id", "--id", dest="id", default="unique_id")
    parser.add_option("--ITS", "--ITS", dest="ITS", default=500)
    parser.add_option("--sim", "--sim", dest="sim", default="N")
    parser.add_option("--file1", "--file1", dest="file1")
    parser.add_option("--file2", "--file2", dest="file2")
    parser.add_option("--profile", "--profile", dest="profile")
    parser.add_option("--LD_file", "--LD_file", dest="LD_file")
    parser.add_option("--particles", "--particles", dest="particles", default=100)
    (options, args) = parser.parse_args()

    # set seed
    seed = float(options.seed)
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    # get core experiment values
    M = int(options.M)
    N1 = int(options.N1)
    N2 = int(options.N2)
    Ns_sim = int(options.Ns_sim)
    Ns = options.Ns
    if Ns is not None:
        Ns = int(Ns)
    ITS = int(options.ITS)
    particles = int(options.particles)

    # if user provides known heritabilies and/or gen-corr and/or known proportions 
    H1 = options.H1
    H2 = options.H2
    rho = options.rho
    A00 = options.A00
    A10 = options.A10 
    A01 = options.A01
    A11 = options.A11 

    if A00 is not None and A10 is not None and A01 is not None and A11 is not None:
        A00_true = float(A00)
        A10_true = float(A10)
        A01_true = float(A01)
        A11_true = float(A11)
    else:
        # user did not provide known proportions 
        A00_true = None 
        A10_true = None
        A01_true = None 
        A11_true = None 

    if H1 is not None:
        H1_true = float(H1)
    else:
        # user did not provide known heritability
        H1_true = None

    if H2 is not None:
        H2_true = float(H2)
    else:
        # user did not provide known heritability
        H2_true = None

    if rho is not None:
        rho_true = float(rho)
        true_corr = rho_true 
    else:
        # user did not provide known gen corr
        rho_true = None


    # simulate effect sizes
    file1 = options.file1
    file2 = options.file2
    sim = options.sim

    # if simulating with LD
    LD_file = options.LD_file
    if LD_file is not None:
        V = np.loadtxt(LD_file)
        # truncate
        V[:] = truncate_matrix(V)

    else:
        V = None

    if sim == "Y":
        if options.H1_sim == None or options.H2_sim == None or options.rho_sim == None:
            print "Error: need to specify H1, H2, rho when simulating!"
            exit(0)
        A00_sim = float(options.A00_sim)
        A10_sim = float(options.A10_sim)
        A01_sim = float(options.A01_sim)
        A11_sim = float(options.A11_sim)
        H1_sim = float(options.H1_sim)
        H2_sim = float(options.H2_sim)
        rho_sim = float(options.rho_sim)
        rho_e_sim =float(options.rho_e_sim)

        z1, z2, true_corr = simulate(A00_sim, A10_sim, A01_sim, A11_sim, H1_sim, H2_sim, rho_sim, rho_e_sim, M, N1, N2, Ns_sim, V)
        print "True corr: %.4f" % true_corr 
        if rho is not None:
            true_corr = rho_true
    else:
        if file1 is None and file2 is  None:
            print "Error: Need sumstats file!"
            sys.exit(0)
        else:
            z1, z2 = load_sumstats(file1, file2)
            A00_sim = None 
            A10_sim = None
            A01_sim = None
            A11_sim = None
            H1_sim = None
            H2_sim = None
            rho_sim = None
            rho_e_sim =None

    # print UNITY header
    ID = options.id
    f = open("out.%s.%d.p%d" % (ID, seed, particles), 'w')

    profile = options.profile

    # profile code for time-benchmarking
    if profile == "Y":
        pr = cProfile.Profile()
        pr.enable()

    # debugging
    #print z1
    #print z2

    # STEP 1: get estimates for H1, H2, and rho

    # no LD
    if V is None and H1_true is None and H2_true is None and rho_true is None: # joint estimation  
        p00_est, p10_est, p01_est, p11_est, H1_est, H2_est, rho_est, cov_e_coef_est \
            = initial_estimates(N1, N2, M, z1, z2, "L-BFGS-B")
    elif V is None and H1_true is not None and H2_true is not None and rho_true is not None: 
        print "Optimizing only p parameter..." 
        p00_est, p10_est, p01_est, p11_est, H1_est, H2_est, rho_est, cov_e_coef_est \
            = initial_estimates(N1, N2, M, z1, z2, "L-BFGS-B", H1=H1_true, H2=H2_true, rho=rho_true)
    else: # there's LD
        #p00_est, p10_est, p01_est, p11_est, H1_est, H2_est, rho_est, cov_e_coef_est \
        #    = initial_estimates(N1, N2, M, z1, z2, "L-BFGS-B", H1_true, H2_true, rho_true, V)
        p00_est = .25
        p10_est = .25
        p01_est = .25
        p11_est = .25

        H1_est = H1_true
        H2_est = H2_true
        rho_est = rho_true
        cov_e_coef_est = 0

    # set cov_e coef estimate so can use in density estimate
    if Ns == 0:  # if user knows there's no sample overlap, set to zero
        cov_e_coef = 0
    else:  # otherwise, use MAP estimate
        cov_e_coef = cov_e_coef_est

    # print header
    print "- - - - - - - - - - UNITY - - - - - - - - -"
    f.write("\n- - - - - - - - - - UNITY - - - - - - - - -\n")

    print_header(N1, N2, Ns_sim, Ns, M, ITS, A00_true, A10_true, A01_true, A11_true, A00_sim, A10_sim, A01_sim, A11_sim, H1_true, H2_true, H1_sim, H2_sim,
                 rho_true, rho_sim, true_corr, rho_e_sim, sim, file1, file2, ID, f, LD_file)

    # find true density
    if sim == "Y":
        params = [[A00_sim, A10_sim, A01_sim, A11_sim], H1_sim, H2_sim, rho_sim, cov_e_coef]
        try:
            true_MAP = log_p_pdf_fast(z1, z2, params, M, N1, N2)
        except:
            print "error: cannot calculate posterior density due to non-pos-sem-def" 

        true_log_like = log_likelihood(z1, z2, params, M, N1, N2)

        if V is None:
            print "True like: %.4f" % true_log_like
            print "True MAP: %.4f" % true_MAP

            f.write("True likelihood: %.4f\n" % true_log_like)
            f.write("True posterior-density: %.4f\n" % true_MAP)
        else: # LD-version
            print "True like: "
            #true_MAP = true_map_ld(z1, z2, [A00, A10, A01, A11], H1_sim, H2_sim, rho_sim, cov_e_coef, M, N1, N2, V)
            #true_log_like = true_like_ld(z1, z2, [A00, A10, A01, A11], H1_sim, H2_sim, rho_sim, cov_e_coef, M, N1, N2, V)
            #print "True like LD: %.4f" % true_log_like
            #print "True MAP LD: %.4f" % true_MAP

            #f.write("True likelihood: %.4f\n" % true_log_like)
            #f.write("True posterior-density: %.4f\n" % true_MAP)
    print "- - - - - - - - - - - - - - - - - - - - -"
    f.write("- - - - - - - - - - - - - - - - - - - - -\n")

    f.write("\n")


    # print results from Step 1
    print "\n- - - - - Step 1: MAP estimates - - - - -"
    f.write("\n- - - - - Step 1: MAP estimates - - - - -\n")

    f.write("Step 1 estimated...\n")
    print "p00 MAP-estimate: %.6g" % p00_est
    print "p10 MAP-estimate: %.6g" % p10_est
    print "p01 MAP-estimate: %.6g" % p01_est
    print "p11 MAP-estimate: %.6g" % p11_est
    print "H1 MAP-estimate: %.4g" % H1_est
    print "H2 MAP-estimate: %.4g" % H2_est
    print "rho MAP-estimate: %.4g" % rho_est
    print "cov_e_coef MAP-estimate: %.4g" % cov_e_coef_est

    f.write("p00 MCMC-estimate: %.6g \n" % p00_est)
    f.write("p10 MCMC-estimate: %.6g \n" % p10_est)
    f.write("p01 MCMC-estimate: %.6g \n" % p01_est)
    f.write("p11 MCMC-estimate: %.6g \n" % p11_est)
    f.write("H1 MCMC-estimate: %.4g \n" % H1_est)
    f.write("H2 MCMC-estimate: %.4g \n" % H2_est)
    f.write("rho MCMC-estimate: %.4g \n" % rho_est)
    f.write("cov_e_coef MCMC-estimate: %.4g\n" % cov_e_coef_est)

    print "- - - - - - - - - - - - - - - - - - - - -"
    f.write("- - - - - - - - - - - - - - - - - - - - -\n")

    # if user gave H1, H2, rho
    if H1_true is None:
        H1_0 = H1_est
    else:
        H1_0 = H1_true 

    if H2_true is None:
        H2_0 = H2_est
    else:
        H2_0 = H2_true 

    if rho_true is None:
        rho_0 = rho_est
    else:
        rho_0 = rho_true 

    if A00_true is None:
        p00_0 = p00_est
    else:
        p00_0 = A00_true 

    if A10_true is None:
        p10_0 = p10_est 
    else:
        p10_0 = A10_true 

    if A01_true is None:
        p01_0 = p01_est 
    else:
        p01_0 = A01_true 

    if A11_true is None:
        p11_0 = p11_est 
    else:
        p11_0 = A11_true 


    # STEP 2: run MCMC to estimate p-vec

    # new file for chain readings
    f_chain = open("chain.%s.%d.p%d" % (ID, seed, particles), 'w')

    # run Step 2: MCMC
    init_values = [p00_0, p10_0, p01_0, p11_0, H1_0, H2_0, rho_0, cov_e_coef]

    if V is None: # no LD, do regular MCMC
        print "Running MCMC with rho: %.4f" % rho_true 
        p00, p10, p01, p11, H1, H2, rho, a00_std, a10_std, a01_std, a11_std, H1_std, H2_std, rho_std, rho_first_quantile, rho_third_quantile \
        = run_MCMC(init_values, N1, N2, M, z1, z2, ITS, A00_true, A10_true, A01_true, A11_true, H1_true, H2_true, rho_true, f_chain)

    # print estimates for Step 2: MCMC
    # print results from Step 1
    print "\n- - - - - Step 2: MCMC estimates - - - - -"
    f.write("\n- - - - - Step 2: MCMC estimates - - - - -")

    print "mcmc-p00: %.6g" % p00
    print "mcmc-p10: %.6g" % p10
    print "mcmc-p01: %.6g" % p01
    print "mcmc-p11: %.6g" % p11
    print "mcmc-H1: %.4g" % H1
    print "mcmc-H2: %.4g" % H2
    print "mcmc-rho: %.4g" % rho

    f.write('\n')
    f.write("mcmc-p00: %.6g \n" % p00)
    f.write("mcmc-p10: %.6g \n" % p10)
    f.write("mcmc-p01: %.6g \n" % p01)
    f.write("mcmc-p11: %.6g \n" % p11)
    f.write("mcmc-H1: %.4g \n" % H1)
    f.write("mcmc-H2: %.4g \n" % H2)
    f.write("mcmc-rho: %.4g \n" % rho)

    print "p00_std: %.6g" % a00_std
    print "p10_std: %.6g" % a10_std
    print "p01_std: %.6g" % a01_std
    print "p11_std: %.6g" % a11_std

    if H1_true is None and H2_true is None and rho_true is None:
        print "H1_std: %.6g" % H1_std
        print "H2_std: %.6g" % H2_std
        print "rho_std: %.6g" % rho_std

    f.write('\n')
    f.write("p00_std: %.6g\n" % a00_std)
    f.write("p10_std: %.6g\n" % a10_std)
    f.write("p01_std: %.6g\n" % a01_std)
    f.write("p11_std: %.6g\n" % a11_std)


    f.write("H1_std: %.6g\n" % H1_std)
    f.write("H2_std: %.6g\n" % H2_std)
    f.write("rho_std: %.6g\n" % rho_std)

    # rho percentiles 
    print "rho-2.5-percentile: %.4g" % rho_first_quantile
    print "rho-97.5-percentile: %.4g" %  rho_third_quantile
    print "rho-true: %.4g" % true_corr

    f.write("rho-2.5-percentile: %.4g\n" % rho_first_quantile)
    f.write("rho-97.5-percentile: %.4g\n" %  rho_third_quantile)
    f.write("rho-true: %.4g\n" % true_corr)

    # print desnsity 
    a=[(1-p10-p01-p11), p10, p01, p11]
    params = [a, H1, H2, rho, 0]
    try:
        MAP=log_p_pdf_fast(z1, z2, params, M, N1, N2)
    except: 
        MAP = 0 
    #print "Final density: %.4f" % MAP
    #f.write("Final density: %.4f\n" % MAP)

    f.write("- - - - - - - - - - - - - - - -  - - - - -\n")
    print "- - - - - - - - - - - - - - - -  - - - - -"

    # close out and chain files
    f.close()
    f_chain.close()

    if profile == "Y":
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

    sys.exit(0)


if __name__ == "__main__":
    main()

