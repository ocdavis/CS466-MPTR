"""
Vignette to demonstrate the capabilities of the tribal package.
"""

from tribal.preprocess import preprocess
from tribal import df, roots
from tribal import Tribal
import time


if __name__ == "__main__":


            
    isotypes = ['IGHM', 'IGHG3', 'IGHG1', 'IGHA1','IGHG2','IGHG4','IGHE','IGHA2']

    #test dnapars
    clonotypes, df_filt = preprocess(df, roots,isotypes, cores=3, verbose=True )

    #test tribal
    tr = Tribal(n_isotypes=len(isotypes), verbose=True, restarts=1, niter=15)
            
    #run in refinement mode
    start = time.time()
    shm_score, csr_likelihood, best_scores, transmat = tr.fit(clonotypes=clonotypes, mode="refinement", cores=6,ilp_type=1)
    end = time.time()

    total = end - start

    print("Took " + str(total) + " seconds to run.")

