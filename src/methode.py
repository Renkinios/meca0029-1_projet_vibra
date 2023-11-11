import numpy as np

def methode_superposition(M,K,w,x,eps,p,t,nMode):
    """ Cree les modes de deplasament par rapport a la methode de superposition, utilisation
    d'une fonction pour regarder plus facilement la convergence de celle-ci par rapport a nMode
        Arguments : 
            - M : Matrice de Masse
            - K : Matrice de raideur
            - w : vitesse angulaire
            - x : vecteur propre
            - eps : damping ratio
            - p : force appliquee
            - t : temps
            - nMode : nombre de mode repris 
        Return : 
            - param : q_deplacement, q_acc 
    """
    w_d = w * np.sqrt(1-eps**2)
    phi = np.zeros((nMode,len(t)))
    n = np.zeros((nMode,len(t)))
    A = 0
    B = 0
    x = x.real[:nMode]
    for r in range(nMode) :
        mu     = x[r].T@M@x[r]
        phi[r] = x[r].T@p/mu
        h      = 1/w[r]*np.exp(-eps[r]*w[r]*t)*np.sin(w_d[r]*t)
        n[r]   = np.exp(-eps[r]*w[r]*t)*(A*np.cos(w_d[r]*t) + B*np.sin(w_d[r]*t)) + np.convolve(phi[r],h,mode="same")  #* (t[1]-t[0])  #pas sur le dernier terme
    q_deplacement = n.T@x # T*M
    phi_w = np.zeros((nMode,len(t)))
    for i in range(nMode) : 
        phi_w[i] = phi[i]/w[i]
    q_acc = q_deplacement + (np.linalg.inv(K)@p).T -phi_w.T@x
    return q_deplacement, q_acc