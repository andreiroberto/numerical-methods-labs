# Nume student: Stoica Andrei-Roberto
# MP - Metoda puterii
# MPI - Metoda puterii inverse
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
#Initializari:
n = 6
A1 = rng.integers(1,99, size=(n,n))#Generarea matricei fara valoare proprie dominanta cu transformare asemenea
T = rng.integers(1,99, size=(n,n))
D= np.diag([2,2,2,2,2,2])
A2 = T@D@np.linalg.inv(T)#Generearea matricei cu valoarea proprie dominanta 
maxiter=1000
tol = 1e-3

def MP(A,n):
    y=A[:,0]#Prima coloana din matrice
    y=np.divide(y,np.linalg.norm(y))
    i=0
    e = np.zeros((n,1))
    e[i]=1
    while e[i] > tol:
        if i > maxiter:
            print('Numarul maxim de iteratii a fost atins,nivelul prescris al tolerantiei nu a fost atins /eroare.\n')
            break
        z=A@y
        z=np.divide(z,np.linalg.norm(z))
        e[i+1]=abs(1-abs(np.transpose(z)@y))
        y=z
        i+=1
    lamda=np.divide(A@y,y)[0]#Prima valoare proprie este si cea dominanta
    e[0]=0
    return z, lamda, e

def MPI(A,n):
    y=A[:,0]#Prima coloana din matrice
    y=np.divide(y,np.linalg.norm(y))
    i=0
    e = np.zeros((n,1))
    e[i]=1
    while e[i] > tol:
        if i > maxiter:
            print('S-a atins numarul maxim de iteratii fara a se fi obtinut nivelul prescris al tolerantei.\n')
            break   
        mu=y@A@y#Se calculeaza deplasarea
        z=np.linalg.solve(mu*np.eye(n)-A,y)#Se rezolva sistemul liniar
        z=np.divide(z,np.linalg.norm(z))
        e[i+1]=abs(1-abs(np.transpose(z)@y))
        y=z
        i+=1
    lamda=np.transpose(np.power(y,i-2))@A@np.power(y,i-2)#Valoarea proprie dupa cea mai buna deplasare cu ajutorul catului Rayleigh
    e[0]=0
    return z, lamda, e

#Aplicarea metodelor
z11, lamda11, e11 = MP(A1,n)
z12, lamda12, e12 = MPI(A1,n)
z21, lamda21, e21 = MP(A2,n)
z22, lamda22, e22 = MPI(A2,n)
#Verificari:
#print('-----------z------------\n',z12)
#print('-----------lamda------------\n',lamda12)
#print (np.linalg.eig(A1))

i=np.linspace(0, 5, num=6)
#Graficul 1
plt.subplot(3, 1, 1)
plt.plot(i,e11,e12)
plt.legend(['MP','MPI'])
plt.title("Pentru matricea fara valoare proprie dominanta: ")
plt.ylabel('Eroare')
plt.xlabel('Iteratii')
#Graficul 2
plt.subplot(3, 1, 3)
plt.plot(i,e21,e22)
plt.title("Pentru matricea cu valoare proprie dominanta: ")
plt.legend(['MP','MPI'])
plt.ylabel('Eroare')
plt.xlabel('Iteratii')

plt.show()