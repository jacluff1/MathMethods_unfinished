import os
import numpy as np
import pdb

def directory_checker(dirname):
    """ check if a directory exists, creates it if it doesn't"""
    dirname =   str(dirname)
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except:
            os.stat(dirname)

def find_lowest(F,N_lowest):
    """ find the index values m,n for the first 'N' lowest frequencies """
    m_max   =   len(F[:,0])
    n_max   =   len(F[0,:])
    M   =   np.arange(m_max)
    N   =   np.arange(n_max)

    F1  =   F.flatten()
    F1  =   F1.argsort()[:N_lowest]
    F1  =   divmod( F1 , n_max )

    Mi,Ni   =   F1[0],F1[1]
    F       =   np.array([ F[Mi[i],Ni[i]] for i in range(N_lowest) ])

    return Mi,Ni,F
