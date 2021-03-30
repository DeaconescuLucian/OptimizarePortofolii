import numpy as np
import matplotlib.pyplot as grafic
from functii.teste2 import generate_matrix
from math import e

def citeste_date(nume):
    R=np.genfromtxt(nume)
    #R=generate_matrix()
    #print(R)
    nr_obs=R.shape[0]
    nr_actiuni=R.shape[1]
    B=-np.ones([nr_actiuni,nr_actiuni-1])
    B[:nr_actiuni-1,:nr_actiuni-1]=np.eye(nr_actiuni-1)
    alpha=np.zeros([nr_actiuni,1])
    alpha[nr_actiuni-1]=1
    rmed=np.mean(R,axis=0)
    rmed=np.reshape(rmed,[nr_actiuni,1])
    #matricea de covarianta - estimatie nedeplasata
    Q=np.cov(R.T)*(nr_obs-1)/nr_obs
    return Q,rmed,alpha,B, nr_actiuni

def fobiectiv(Q,rmed,B,alpha,ro,Rp,x):
    y=alpha+np.matmul(B,x)
    #np.matmul(a,b)=a*b - inmultirea a doua matrice
    y1=np.matmul(y.T,Q)
    V=np.matmul(y1,y)
    c=ro/(Rp*Rp)
    y=np.matmul(rmed.T,alpha)
    y1=np.matmul(rmed.T,B)
    y2=np.matmul(y1,x)
    val=V+c*np.square(y-Rp+y2)
    return val,V

def gen_firefly(nr_actiuni):
    firefly=np.zeros(nr_actiuni-1)
    s=0
    i=0
    generated=np.zeros(nr_actiuni-1)
    while i<nr_actiuni-1 and s<1:
        value=np.random.uniform(0,1)
        j = np.random.randint(0, nr_actiuni - 1)
        while generated[j]:
            j = np.random.randint(0, nr_actiuni - 1)
        if value+s<=1:
            generated[j]=1
            firefly[j]=value
            s=s+value
        else:
            firefly[j] = 1-s
            s=1
        i+=1
    return firefly


def gen_pop(dim,nr_actiuni,Q,rmed,B,alpha,ro,Rp):
    pop=np.zeros([dim,nr_actiuni])
    for i in range(dim):
        x=gen_firefly(nr_actiuni)
        pop[i][:nr_actiuni-1]=x
        ps=x[:nr_actiuni-1]
        y=ps.reshape([nr_actiuni-1,1])
        val,V=fobiectiv(Q,rmed,B,alpha,ro,Rp,y)
        pop[i][nr_actiuni - 1]=1/(1+val)
    return pop

def move_firefly(better,worse,nr_actiuni,Q,rmed,B,alpha,ro,Rp,gamma,attractiveness,randomization_factor):
    RHS = np.zeros(nr_actiuni-1)
    random = np.zeros(nr_actiuni-1)
    for i in range(nr_actiuni-1):
        random[i] = randomization_factor*np.random.uniform(-1, 1)
    for i in range(nr_actiuni-1):
        RHS[i] = attractiveness*pow(e, -gamma*pow(worse[i] - better[i], 2)) * (better[i] - worse[i])
    intermediar = np.add(worse[:nr_actiuni-1], RHS)
    rez = np.add(intermediar, random)
    for i in range(nr_actiuni-1):
        if rez[i]<0:
            rez[i]=0
    if np.sum(rez) > 1:
        sum = np.sum(rez)-1
        while sum>0:
            poz=np.random.randint(nr_actiuni-1)
            if rez[poz]>sum:
                rez[poz]=rez[poz]-sum
                sum=0
            else:
                sum = sum - rez[poz]
                rez[poz]=0
    worse[:nr_actiuni - 1] = rez
    y=rez.reshape([nr_actiuni-1,1])
    val, V = fobiectiv(Q, rmed, B, alpha, ro, Rp, y)
    worse[nr_actiuni-1]=1/(1+val)
    return worse

def sortare_populatie(pop,nr_actiuni,dim):
    for i in range(dim-1):
        for j in range(i,dim,1):
            if pop[i][nr_actiuni-1]<pop[j][nr_actiuni-1]:
                x=pop[i][nr_actiuni-1]
                pop[i][nr_actiuni - 1]=pop[j][nr_actiuni-1]
                pop[j][nr_actiuni - 1]=x
    return pop

def arata(sol,n,v,Q, rmed, B, alpha, ro, Rp):
    # vizualizare rezultate RISCMIN1M
    x=sol.reshape([n,1])
    val,V=fobiectiv(Q, rmed, B, alpha, ro, Rp, x)
    t=len(v)
    print("Cea mai buna valoare calculată RiscMin1M: ",val)
    print("Riscul minim",V)
    sol_c=np.zeros(n+1)
    sol_c[:n]=sol
    sol_c[n]=1-sum(sol)
    print("Alegerea corespunzatoare este: ",sol_c)
    Rr = np.matmul((rmed.T),(alpha+np.matmul(B,x)))
    print("Randamentul obtinut: ",Rr)
    fig=grafic.figure()
    x=[i for i in range(t)]
    y=[(1-v[i])/v[i] for i in range(t)]
    grafic.plot(x,y,'ro-')
    grafic.ylabel("Valoarea")
    grafic.xlabel("Generația")
    grafic.title("Evoluția calității celui mai bun individ din fiecare generație")
    grafic.show()

def HSFA():
    dim=50
    ro=10
    Rp=2
    maxGen = 500
    t=0
    Q, rmed, alpha, B, nr_actiuni = citeste_date("portofoliu1.txt")
    pop = gen_pop(dim, nr_actiuni, Q, rmed, B, alpha, ro, Rp)
    pop=sortare_populatie(pop,nr_actiuni,dim)
    history=np.zeros([maxGen+1,nr_actiuni])
    history[t]=pop[0]
    keep=dim//10
    gamma=1
    randomization_factor=2
    HMCR=0.9
    PAR=0.1
    attractiveness=1
    while t<maxGen:
        for i in range(keep):
            for j in range(dim):
                if pop[j][nr_actiuni-1]<pop[i][nr_actiuni-1]:
                    pop[j]=move_firefly(pop[i],pop[j],nr_actiuni,Q,rmed,B,alpha,ro,Rp,gamma,attractiveness,randomization_factor)
                else:
                    for k in range(nr_actiuni-1):
                        rand = np.random.uniform(0, 1)
                        if rand<HMCR:
                            r1=int(dim*rand)
                            pop[i][k]=pop[r1][k]
                            rand1=np.random.uniform(-1,1)
                            if rand1<PAR:
                                pop[i][k]=pop[i][k]+rand1/2
                        else:
                            pop[i][k]=pop[dim-1][k]+rand*(pop[0][k]-pop[dim-1][k])
                        if pop[i][k]<0:
                            pop[i][k]=0
                    if np.sum(pop[i][:nr_actiuni-1])>1:
                        sum = np.sum(pop[i][:nr_actiuni-1]) - 1
                        while sum > 0:
                            poz = np.random.randint(nr_actiuni - 1)
                            if pop[i][:nr_actiuni-1][poz] > sum:
                                pop[i][:nr_actiuni-1][poz] = pop[i][:nr_actiuni-1][poz] - sum
                                sum = 0
                            else:
                                sum = sum - pop[i][:nr_actiuni-1][poz]
                                pop[i][:nr_actiuni-1][poz] = 0
                    ps = pop[i][:nr_actiuni - 1]
                    y = ps.reshape([nr_actiuni - 1, 1])
                    val, V = fobiectiv(Q, rmed, B, alpha, ro, Rp, y)
                    pop[i][nr_actiuni-1]=1/(1+val)
        pop=sortare_populatie(pop,nr_actiuni,dim)
        randomization_factor = randomization_factor * 0.96
        if history[t][nr_actiuni-1]>=pop[0][nr_actiuni-1]:
            history[t+1]=history[t]
        else:
            history[t + 1] = pop[0]
        t+=1

    print(history[len(history)-1][nr_actiuni-1])
    x = history[len(history)-1][:nr_actiuni-1]
    history_v=np.zeros(t)
    for i in range(t):
        history_v[i]=history[i][nr_actiuni-1]
    print("sol")
    print(x)
    print("istoric_v")
    print(history_v)
    arata(x, nr_actiuni - 1, history_v, Q, rmed, B, alpha, ro, Rp)
HSFA()









