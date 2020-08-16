import numpy as np
import sys

def contraction(p,q,r,s):
    if np.abs(p-q) != 1 or np.abs(r-s) != 1:
        return 0
    elif max(p,q)%2 == 0 or max(r,s)%2 == 0:
        return 0
    else:
        if q == p+1 and s == r+1:
            return 1
        elif q == p+1 and s == r-1:
            return -1
        elif q == p-1 and s == r+1:
            return -1
        elif q == p-1 and s == r-1:
            return 1
        else:
            return 0

def init_channels(h,p,xi,g):
    f_ij = np.zeros((h,h))
    f_ab = np.zeros((p,p))
    V_ijkl = np.zeros((h,h,h,h))
    V_abcd = np.zeros((p,p,p,p))
    V_abij = np.zeros((p,p,h,h))

    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    kronecker = contraction(a,b,c,d)
                    V_abcd[a,b,c,d] = -0.5*g*kronecker
                    if a < h and b < h and c < h and d < h:
                        V_ijkl[a,b,c,d] = -0.5*g*kronecker
                    if c < h and d < h:
                        V_abij[a,b,c,d] = -0.5*g*kronecker

    # Fock matrices
    for p in range(p+h):
        for q in range(p+h):
            if p < h and q < h:
                if p == q:
                    f_ij[p,q] = xi*(p - p%2)/2
                    for i in range(h):
                        f_ij[p,q] += -0.5*g*contraction(p,i,q,i)
            if p >= h and q >= h:
                if p == q:
                    f_ab[p-h,q-h] = xi*(p - p%2)/2


    return f_ij,f_ab,\
            V_ijkl,V_abcd,V_abij

def init_cluster(h,p,f_ij,f_ab,V_abij):
    t = np.zeros((p,p,h,h))
    t_new = np.zeros((p,p,h,h))
    D_abij = np.zeros((p,p,h,h))

    for a in range(p):
        for b in range(p):
            for i in range(h):
                for j in range(h):
                    D_abij[a,b,i,j] = f_ij[i,i] + f_ij[j,j] - f_ab[a,a] - f_ab[b,b]
                    if D_abij[a,b,i,j] == 0:
                        print(f_ij[i,i],f_ij[j,j],f_ab[a,a],f_ab[b,b],V_abij[a,b,i,j])
                    t[a,b,i,j] = V_abij[a,b,i,j]/D_abij[a,b,i,j]
    return t,t_new,D_abij

def update_matrix_elements(h,p,t,f_ij,f_ab,V_abcd,V_ijkl,V_abij,X_ijkl,X_il,X_ad,X_bcjk):
    H_abij = np.zeros((p,p,h,h))

    term1 =      V_abij
    term2 =      np.einsum('bc,acij->abij',f_ab,t) - np.einsum('ac,bcij->abij',f_ab,t)
    term3 =      np.einsum('kj,abik->abij',f_ij,t) - np.einsum('ki,abjk->abij',f_ij,t)
    term4 = 0.5* np.einsum('abcd,cdij->abij',V_abcd,t)
    term5 = 0.5* np.einsum('klij,abkl->abij',V_ijkl,t)
    term6 = 0.5*(np.einsum('bcjk,acik->abij',X_bcjk,t)\
               - np.einsum('acjk,bcik->abij',X_bcjk,t)\
               - np.einsum('bcik,acjk->abij',X_bcjk,t)\
               + np.einsum('acik,bcjk->abij',X_bcjk,t))
    term7 = 0.5*(np.einsum('il,ablj->abij',X_il,t) - np.einsum('jl,abli->abij',X_il,t))
    term8 = 0.5*(np.einsum('ad,dbij->abij',X_ad,t) - np.einsum('bd,daij->abij',X_ad,t))
    term9 = 0.25*(np.einsum('ijkl,abkl->abij',X_ijkl,t))

    H_abij = term1 + term2 -term3 + term4 + term5 + term6 + term7 + term8 + term9
    return H_abij

def update_cluster(h,p,t,t_new,H_abij,D_abij,alpha): 
    for a in range(p):
        for b in range(p):
            for i in range(h):
                for j in range(h):
                    t_new[a,b,i,j] = t[a,b,i,j] + H_abij[a,b,i,j]/D_abij[a,b,i,j]
    t = alpha*t_new + (1-alpha)*t
    return t,t_new

def update_intermediates(h,p,t,V_klcd):
    X_ijkl = np.einsum('cdij,klcd->ijkl',t,V_klcd)
    X_il = -np.einsum('cdki,klcd->il',t,V_klcd)
    X_ad = np.einsum('ackl,klcd->ad',t,V_klcd)
    X_bcjk = np.einsum('dblj,klcd->bcjk',t,V_klcd)
    return X_ijkl,X_il,X_ad,X_bcjk

def energy(h,p,V_klcd,t):
    E = 0
    for a in range(p):
        for b in range(p):
            for i in range(h):
                for j in range(h):
                    E += V_klcd[i,j,a,b]*t[a,b,i,j]
    return 0.25*E

def reference_energy(h,p,xi,g):
    E = 0
    for i in range(h):
        E += xi*(i - i%2)/2
        for j in range(h):
            E += -0.25*g*contraction(i,j,i,j)
    return E

def run(h,p,xi,g,max_it=500,eps=1e-10,alpha=1):
    f_ij,f_ab,V_ijkl,V_abcd,V_abij = init_channels(h,p,xi,g)
    t,t_new,D_abij = init_cluster(h,p,f_ij,f_ab,V_abij)
    E_ref = reference_energy(h,p,xi,g)
    E1 = 0
    for step in range(max_it):
        X_ijkl,X_il,X_ad,X_bcjk = update_intermediates(h,p,t,V_abij.T)
        H_abij = update_matrix_elements(h,p,t,f_ij,f_ab,V_abcd,V_ijkl,V_abij,X_ijkl,X_il,X_ad,X_bcjk)
        t,t_new = update_cluster(h,p,t,t_new,H_abij,D_abij,alpha)
        E2 = energy(h,p,V_abij.T,t)
        if np.abs(E1-E2) < eps:
            print('Converged after step:',step)
            break
        E1 = E2
    return E2,E_ref

def CC_ground(h,p,xi,g,max_it=500,eps=1e-10,alpha=1):
    Ec,Eref = run(h,p,xi,g,max_it=max_it,eps=eps,alpha=alpha)
    E = Eref + Ec
    return E,Eref,Ec

if __name__ == '__main__':
    h = int(sys.argv[1])
    p = int(sys.argv[2])
    xi = float(sys.argv[3])
    g = float(sys.argv[4])
    try:
        alpha = float(sys.argv[5])
    except IndexError:
        alpha = 0.5
    print('CCD calculations of pairing model:')
    print('\t {}h - {}p'.format(h,p))
    print('\t g  = {}'.format(g))
    print('\t xi = {}'.format(xi))
    E2,E_ref = run(h,p,xi,g,alpha=alpha)
    print('Reference Energy (E_ref): ',E_ref)
    print('Correlation Energy (E_cc):',E2)
    print(' -> Total energy:',E2+E_ref)
