import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from numba import jit
import math
from numba import jit, cuda
from PIL import Image
 
import warnings
warnings.filterwarnings('ignore')



def init(deltaM,r_D,metoile,dr,Mr): # initialisation des positions
    
    
    x=np.random.normal(0.,r_D,N)
    y=np.random.normal(0.,r_D,N)
    rayon=np.sqrt(x**2+y**2)
    iann=(rayon/dr).astype(int) # anneau ou tombe chaque particule
    for k in range(0,N): 
        deltaM[iann[k]]+=metoile # incremente masse dans cet anneau
    Mr[0]=deltaM[0]
    return x,y,rayon,iann,deltaM,Mr

'''
@jit
def vrott(vrota,x,y,r_D,dr,vx,vy): # initialisation des positions

    cos=np.zeros(x.size)
    sin=np.zeros(y.size)
    vr=np.zeros(x.size)
    vrotmoy = np.zeros(N_a)
    No      = np.zeros(N_a)
    rayon=np.sqrt(x**2+y**2)
    sin=y/rayon
    cos=x/rayon
    vr=-vx*sin+vy*cos 
    for k in range(0,N): 
        
        vrotmoy[iann[k]] += vr[k]
        No[iann[k]] += 1

    indexes = np.where(No > 0)
    vrotmoy[indexes] /= No[indexes]

    nan_indexes = np.where(No == 0)
    vrotmoy[nan_indexes] = np.nan
        
    
    return vrotmoy

    '''
def calc_vrot_r(x,y,vx,vy,L,N_a,dr):
    # FONCTION CALCULANT LA COURBE DE ROTATION
    # Calcul de la vitesse rotationnelle moyenne de chaque anneau
    r_a = np.linspace(0, L, N_a) # rayon de chaque anneau
    iann=(rayon/dr).astype(int) # indice anneau pour chaque etoile
    r = np.sqrt(x**2 + y**2) # rayon de chaque etoile
    vrot=-vx*y/r + vy*x/r # vitesse azimutale de chaque etoile
    vrotMoy=np.zeros(N_a) # tableau vitesse azimutale moyenne
    for i in range(N_a): # boucle sur tous les anneaux
        vrotMoy[i]=np.mean(vrot[np.where(iann==i)]) # moyenne sur anneau i
    # conversion d’unite
    kpc = 30856775814913673 * 1000 # de kpc a m
    P_sol = 7.5e15 # de P_sol a s
    ConversionKmSm1 = (kpc/P_sol)/1000 # Conversion de kpc/P_sol a km/s
    return r_a,vrotMoy*ConversionKmSm1
# END FONCTION CALC_VROT

@jit
def initVit(N,Omega,iann,x,y): # initialisation des vitesses
    vx=np.zeros(N) # vitesse radiale
    vy=np.zeros(N) # vitesse azimutale
    vrot=np.zeros(N)
    
    
    vrot=Omega[iann]*rayon
    vx=-Omega[iann]*y#+np.random.normal(0.,vrot*0.05,N)# orbites circulaires
    vy= Omega[iann]*x#+np.random.normal(0.,vrot*0.05,N) # NB: x/rayon=cos(theta), y/rayon=sin(theta)


    return vx,vy



def massAn(N,N_a,Mr,deltaM,Ggrav,dr): # masse cumulative sous chaque anneau
    rn=(np.arange(1,N_a,1)+1)*dr

    Mr[1:]=np.cumsum(Mr[:-1]+deltaM[1:])+4*np.pi*rn**2*sig0/(1+(rn/rH)**alpha)
    Omega[1:]=np.sqrt(Ggrav*Mr[1:]/(rn)**3)
    Omega[0]=Omega[1]
    return Mr,Omega


@jit
def dens(L,M,x,y,metoile,sigH):
    Delta=2*L/M    
    sigma=np.zeros([M,M]) # densite sur M X M cellules, initialisee a zero
    for n in range(0,N): # boucle sur les N particules
        k=int( (x[n]+L)/Delta ) # numero de cellule en x
        l=int( (y[n]+L)/Delta ) # numero de cellule en y
        sigma[k,l]+=metoile # cumul de la masse dans la cellule
    sigma/=Delta**2 # conversion en densite surfacique
    sigma+=sigH
    return sigma,Delta


@jit
def potentiel(sigma,M,Grav,metoile,Delta):
    pot=np.zeros([M+1,M+1]) # re-initialisation du potentiel
    for k in range(0,M): # boucles sur MXM cellules
        for l in range(0,M):
            if sigma[k,l] > 0.: # cellule ne contribue pas si vide
                for i in range(0,M+1): # boucles sur (M+1)X(M+1) coins
                    for j in range(0,M+1):
                        d=math.sqrt((i-k-0.5)**2+(j-l-0.5)**2) # distance d(i,j,k,l)
                        pot[i,j]+=sigma[k,l]/d # contribution a l’integrale
    pot*=(-Ggrav*metoile*Delta) # les constantes a la fin
    
    return pot

@jit
def force(x,y,pot,Delta,L):
    f_x=np.zeros(N)
    f_y=np.zeros(N)
    ix=np.int_(np.trunc((x+L)/Delta)) # coin de la cellule correspondante
    iy=np.int_(np.trunc((y+L)/Delta))
   
    for n in range(0,N): # boucle sur les particules
        
        f_x[n]=-((pot[ix[n]+1,iy[n]]-pot[ix[n],iy[n]])+(pot[ix[n]+1,iy[n]+1]-pot[ix[n],iy[n]+1]))/(2*Delta)
        f_y[n]=-((pot[ix[n],iy[n]+1]-pot[ix[n],iy[n]])+(pot[ix[n]+1,iy[n]+1]-pot[ix[n]+1,iy[n]]))/(2*Delta)
    
    return f_x,f_y



def Halo(M,L,sig0,rH,alpha):
    xh=np.linspace(-L,L,M)
    yh=np.linspace(-L,L,M)
    Xh,Yh=np.meshgrid(xh,yh)
    Rh=np.sqrt(Xh**2+Yh**2)
    sigH=sig0/(1+(Rh/rH)**alpha)
    return sigH


@jit
def bonce(N,x,y,vx,vy,L):
    L=0.99*L

    x1=2*L-x
    x2=-2*L-x

    y1=2*L-y
    y2=-2*L-y
    for i in range(N):
        

        
        if x1[i]>L:
            vx[i]=-vx[i]
        if x2[i]<-L:
            vx[i]=-vx[i]        
        if y1[i]>L:
            vy[i]=-vy[i]
        if y2[i]<-L:
            vy[i]=-vy[i]     
  
    return vx,vy




Ggrav=2.277e-7 # G en kpc**3/ M_sol / P**2
N=10000# nombre de particules-etoiles
r_D=4 # largeur du disque gaussien, en kpc
metoile=1.e6/(N/1.e5) # masse d’une particule-etoile
x,y,rayon=np.zeros(N),np.zeros(N),np.zeros(N)

N_a=500 # nombre d’anneaux
deltaM=np.zeros(N_a) # masse dans chaque anneau
Mr =np.zeros(N_a) # masse cumulative sous chaque anneau
Omega =np.zeros(N_a) # vitesse angulaire (Keplerienne)
iann =np.zeros(N,dtype='int') # anneau pour chaque particule
dr=6*r_D/N_a # largeur radiale des anneaux
quart=np.random.randint(0,N,int(N/4))

#######################pour la densité ##############################
L=40
M=50
#####################################################################


sig0=1.7490625e8*0
rH=6.28125
alpha=0.2256250

sigh=Halo(M,L,sig0,rH,alpha)









#---------------------------------------------------------------------
sigma,Delta=dens(L,M,x,y,metoile,sigh)


x,y,rayon,iann,deltaM,Mr=init(deltaM,r_D,metoile,dr,Mr)

Mr,Omega=massAn(N,N_a,Mr,deltaM,Ggrav,dr)

vx,vy=initVit(N,Omega,iann,x,y)

pot=potentiel(sigma,M,Ggrav,metoile,Delta)

ii = np.linspace(-L, L, M)
Ix, Iy = np.meshgrid(ii, ii)

fx,fy=force(x,y,pot,Delta,L)
ax=fx/metoile
ay=fy/metoile

xnp1=np.zeros(M)
ynp1=np.zeros(N)

axp1=np.zeros(N)
ayp1=np.zeros(N)

vxnp1=np.zeros(M)
vynp1=np.zeros(N)

ii = np.linspace(-L, L, M)
Ix, Iy = np.meshgrid(ii, ii)

niter=7500*2
dt=0.0002

vrota=np.zeros(N_a)

r=np.linspace(0,N_a*dr,N_a)
for it in range (niter):
    

    sigma,Delta=dens(L,M,x,y,metoile,sigh)

    vx,vy= bonce(N,x,y,vx,vy,L)
    
    xnp1=x+ vx*dt+1/2*ax*dt**2
    ynp1=y+ vy*dt+1/2*ay*dt**2
    
    fxp1,fyp1=force(xnp1,ynp1,pot,Delta,L)
    
    

    
    
    if it%1==0:        
        pot=potentiel(sigma,M,Ggrav,metoile,Delta)
    
    axp1=fxp1/metoile
    ayp1=fyp1/metoile
    
    vxnp1=vx+dt/2*(ax+axp1)
    vynp1=vy+dt/2*(ay+ayp1)
    
    vx=vxnp1
    vy=vynp1
    x=xnp1
    y=ynp1
    ax=axp1
    ay=ayp1
    if it%100==0:
        figit=it//100
        #vrot=vrott(vrota,x,y,r_D,dr,vx,vy)
        fig=plt.figure(1)
        #plt.plot(x[quart],y[quart],'k,')
        plt.plot(x,y,'k,',alpha=0.6)

       # plt.contour(Ix, Iy, np.transpose(pot[0:M,0:M]), colors="red", linewidths=1, zorder=3)        
        plt.xlim(-L,L)
        plt.ylim(-L,L)
        
        plt.title('{:.2}'.format(it*dt),fontsize=30)
        plt.savefig('GraphGalaxie/plot'+str(figit)+'.png') 
        plt.close()
        '''
        plt.figure(2)
        plt.plot(r,vrot,'.')
        plt.savefig('1/plot'+str(it)+'.png')    
        plt.close()
        '''

    print(it)
    
    



frames = []
for i in range(niter//100):
    new_frame = Image.open("GraphGalaxie/plot"+str(i)+".png")
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('png_to_gif.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=50, loop=0)





