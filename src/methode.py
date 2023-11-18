import numpy as np
import functions as fct
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
    for r in range(nMode) :
        X      = x[:,r].real
        mu     = X.T @ M @ X
        phi[r] = X.T @ p/mu
        h      = 1/w_d[r] * np.exp(-eps[r] * w[r] * t) * np.sin(w_d[r] * t)
        convol = np.convolve(phi[r],h)[:len(h)] * (t[1]-t[0])
        n[r]   = np.exp(-eps[r]*w[r]*t)*(A*np.cos(w_d[r]*t) + B*np.sin(w_d[r]*t)) + convol  #pas sur le dernier terme


    q_deplacement = n.T@x[:,:nMode].T
    phi_w = np.zeros((nMode,len(t)))
    for i in range(nMode) : 
        phi_w[i] = phi[i]/(w[i]**2)
    q_acc = q_deplacement + (np.linalg.inv(K)@p).T - phi_w.T@x[:,:nMode].T
    return q_deplacement, q_acc
def New_mth(t,M,C,K,p) : 
    """ Cree les modes de deplasament Newmarks method
        Arguments : 
            - M : Matrice de Masse
            - K : Matrice de raideur
            - p : force appliquee
            - t : temps
            - C : damping matrice
        Return : 
            - param : q 
    """
    q           = np.zeros((len(t),len(M)))
    q_dot       = np.zeros((len(t),len(M)))
    q_2dot      = np.zeros((len(t),len(M)))
    q_star      = np.zeros((len(t),len(M)))
    q_star_dot  = np.zeros((len(t),len(M)))

    #chois gamma et beta corresponds to considering the acceleration average value
    # : q_2dot(tau) = (q_2dot_n + q_2dot_n+1)/2
    gamma = 1/2 
    beta  = 1/4
    h     = t[1]-t[0] #pas de temps
    # condition initial  q_2dot = 0 & q_dot(0) = 0 & q(0) = 0  at the beginning no force
    q[0]          = np.zeros(len(M))
    q_dot[0]      = np.zeros(len(M))
    q_2dot[0]     = np.linalg.inv(M) @ (p[:,0] -C @ q_dot[0] - K @ q[0]) # que des 0 de toute manière
    q_star[0]     = np.zeros(len(M))
    q_star_dot[0] = np.zeros(len(M))
    
    S             = M + gamma * h * C + beta * h**2 * K #retire de l'iteration pour l'optimsation
    S_inverse     = np.linalg.inv(S)                    #retire de l'iteration pour l'optimsation
    for n in range(len(t)-1) : 
        q_star_dot[n+1] = q_dot[n] + (1-gamma) * h * q_2dot[n]
        q_star[n+1]     = q[n] + h * q_dot[n] + (1/2-beta) * h**2 * q_2dot[n]
        q_2dot[n+1]     = S_inverse @ (p[:,n+1].T - C @ q_star_dot[n+1] - K @ q_star[n+1]) 
        q_dot[n+1]      = q_star_dot[n+1] + gamma * h * q_2dot[n+1]
        q[n+1]          = q_star[n+1] + beta * h**2 * q_2dot[n+1]
    return q
def guyan_irons(dof_list,K,M,nMode) :
    """ Methode de guyan_irons
        Arguments : 
            - dof_list : liste des dof
            - K : matrice de raideur
            - M : matrice de masse
        Return : 
            - K_til : matrice de raideur reduite
            - M_til : matrice de masse reduite
    """
    dofR = []
    dofR.append(dof_list[17][:3]) #dof du rotor x,y,z + rotasion z 012 5
    dofR[0].append(dof_list[17][5])
    dofR.append(dof_list[21][:3]) #dof du rotor x,y,z + rotasion z 012 5
    dofR[1].append(dof_list[21][5])
    dofR = np.array([element for sous_liste in dofR for element in sous_liste]) -24 -1 

    Cdofs = np.arange(0, len(K))        
    Cdofs = np.delete(Cdofs, dofR)

    Krr = K[np.ix_(dofR, dofR)]   # retained part
    Krc = K[np.ix_(dofR, Cdofs)]
    Kcc = K[np.ix_(Cdofs, Cdofs)]   # condensed part
    Kcr = K[np.ix_(Cdofs, dofR)]

    Mrr = M[np.ix_(dofR, dofR)]   # retained part
    Mrc = M[np.ix_(dofR, Cdofs)]
    Mcc = M[np.ix_(Cdofs, Cdofs)]   # condensed part
    Mcr = M[np.ix_(Cdofs, dofR)]
    # Guyan-Iron Reduction
    Rgi = np.block([[np.eye(len(Krr))], [-(np.linalg.inv(Kcc) @ Kcr)]])  # transformation matrix
    Kt  = np.block([[Krr, Krc], [Kcr, Kcc]])
    Mt  = np.block([[Mrr, Mrc], [Mcr, Mcc]])
    K_til = Rgi.T @ Kt @ Rgi                       # reduced stiffness matrix
    M_til = Rgi.T @ Mt @ Rgi     
    # K_til = R_gi.T @ K_m @ R_gi
    # M_til = R_gi.T @ M_m @ R_gi
    # w_cc, x_cc = fct.natural_frequency(M_til, K_til,nMode) 
    


    return K_til, M_til, Mcc, Kcc, Rgi, Krr, Kt, Mt

def Craig_Bampton(Mcc,Kcc,Krr,Rgi,Neigenmodes,nMode,Kt,Mt) : 
    """ Methode de Craig_Bampton
        Arguments : 
            - Mcc : matrice de masse condense
            - Kcc : matrice de raideur condense
            - Krr : matrice de raideur reduite
            - Rgi : matrice de transformation
            - Neigenmodes : nombre de mode repris
            - nMode : nombre de mode
            - Kt : matrice de raideur total
            - Mt : matrice de masse total
        Return : 
            - Kcb : matrice de raideur reduite
            - Mcb : matrice de masse reduite
    """
    w_cc,X_cc = fct.natural_frequency(Mcc, Kcc,nMode)
    phi_r = X_cc[:, :Neigenmodes]
    Rcb2 = np.vstack((np.zeros((len(Krr), Neigenmodes)), phi_r)) # transformation submatrix
    Rcb  = np.hstack((Rgi, Rcb2)) # transformation matrix
    Kcb = Rcb.T @ Kt @ Rcb # reduced stiffness matrix
    Mcb = Rcb.T @ Mt @ Rcb # reduced mass matrix
    return Kcb, Mcb
