import numpy as np
from numpy import dot
import sisl as si
from sisl.linalg import inv, solve



def dotA(G, Gamma):
    """ Calculate the spectral function from the Greens function and Gamma """
    return dot(dot(G, Gamma), dagger(G))
def dagger(M):
    """ Take the Hermitian conjugate of a matrix """
    return np.conjugate(M.T)


H = si.get_sile("siesta.TSHS").read_hamiltonian()
geom = si.get_sile("siesta.TSHS").read_geometry()
#print(H, H.shape)
S= si.get_sile("siesta.TSHS").read_overlap()
#print(S,S.shape)


#Hamiltonian dimensions
na = H.na #nr atoms
no = H.no #nr orbitals

print(na,no)

#####################################
#parameters:
# Energies for transmission

Energies = np.arange(-2.2,2.0,0.2)

#SE coupling strength

gammal = 1.0
gammar = 1.0

#SE coupling atoms

couplingatomsL = [0,1,4]
couplingatomsR = [65,66,68]

orbr = H.a2o(couplingatomsR, all=True)
orbl =  H.a2o(couplingatomsL, all=True)
print(orbl)

#####################################



#Set up Gamma matrices
GammaL = np.zeros((no, no))
GammaR = np.zeros((no, no))


  
for j in range(no):
   if j in orbl:
         GammaL[j,j] = gammal
   if j in orbr:
         GammaR[j,j] = gammar
 
print('Gamma set')

k=[0,0,0]

HM = H.Hk(k=k,spin=1,format='array')    
SO = H.Sk(k=k,format='array')


#Transmission calculation
print('Calculating transmission..')
Tfile=open("T.txt","w")
for e in Energies:
        inv_G = (SO * e - HM - 0.5*(GammaL + GammaR)*1j) 
        #print(inv_G.shape)  
        #print("inverting ...")
        G = inv(inv_G)  
        #spectral density
        AL=dotA(G, GammaL)
        AR=dotA(G, GammaR)
        #print GammaL[pvs[0][0],pvs[0][0]]
        TL = np.trace(np.dot(AL,GammaR).real)
        TR = np.trace(np.dot(AR,GammaL).real)
        Tfile.write('{}\t\t{}\t{}\n'.format(e, TL, TR))




