import numpy as np
import matplotlib.pyplot as grafic
from functii.teste2 import generate_matrix
import math

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

def gen_flower(nr_actiuni):
    flower=np.zeros(nr_actiuni-1)
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
            flower[j]=value
            s=s+value
        else:
            flower[j] = 1-s
            s=1
        i+=1
    return flower


def gen_pop(dim,nr_actiuni,Q,rmed,B,alpha,ro,Rp):
    pop=np.zeros([dim,nr_actiuni])
    for i in range(dim):
        x=gen_flower(nr_actiuni)
        pop[i][:nr_actiuni-1]=x
        ps=x[:nr_actiuni-1]
        y=ps.reshape([nr_actiuni-1,1])
        val,V=fobiectiv(Q,rmed,B,alpha,ro,Rp,y)
        pop[i][nr_actiuni - 1]=1/(1+val)
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
    print(v)
    grafic.plot(x,y,'ro-')
    grafic.ylabel("Valoarea")
    grafic.xlabel("Generația")
    grafic.title("Evoluția calității celui mai bun individ din fiecare generație")
    grafic.show()

def draw_Levy_distribution_vector(nr_actiuni,Lambda):
    L=np.zeros(nr_actiuni-1)
    for i in range(nr_actiuni-1):
        gamma = np.random.gamma(Lambda)
        L[i] = (Lambda * gamma * np.sin((math.pi * Lambda) / 2) / math.pi * pow(0.5, 1 + Lambda))
    return L
def draw_from_uniform_distribution(nr_actiuni):
    E=np.zeros(nr_actiuni-1)
    for i in range(nr_actiuni-1):
        E[i]=np.random.uniform(0,1)
    return E

def global_polination(nr_actiuni, Q, rmed, B, alpha, ro, Rp,L,flower,g):
    new_flower=np.zeros(nr_actiuni)
    for i in range(nr_actiuni-1):
        new_flower[i]=flower[i]+L[i]*(g[i]-flower[i])
        if new_flower[i]<0:
            new_flower[i]=0
    if np.sum(new_flower[:nr_actiuni - 1]) > 1:
        sum = np.sum(new_flower[:nr_actiuni - 1]) - 1
        while sum > 0:
            poz = np.random.randint(nr_actiuni - 1)
            if new_flower[:nr_actiuni - 1][poz] > sum:
                new_flower[:nr_actiuni - 1][poz] = new_flower[:nr_actiuni - 1][poz] - sum
                sum = 0
            else:
                sum = sum - new_flower[:nr_actiuni - 1][poz]
                new_flower[:nr_actiuni - 1][poz] = 0
    ps = new_flower[:nr_actiuni - 1]
    y = ps.reshape([nr_actiuni - 1, 1])
    val, V = fobiectiv(Q, rmed, B, alpha, ro, Rp, y)
    new_flower[nr_actiuni-1]=1/(1+val)
    return new_flower

def local_polination(nr_actiuni, Q, rmed, B, alpha, ro, Rp,E,flower,flower1,flower2):
    new_flower = np.zeros(nr_actiuni)
    for i in range(nr_actiuni - 1):
        new_flower[i] = flower[i] + E[i] * (flower1[i] - flower2[i])
        if new_flower[i] < 0:
            new_flower[i] = 0
    if np.sum(new_flower[:nr_actiuni - 1]) > 1:
        sum = np.sum(new_flower[:nr_actiuni - 1]) - 1
        while sum > 0:
            poz = np.random.randint(nr_actiuni - 1)
            if new_flower[:nr_actiuni - 1][poz] > sum:
                new_flower[:nr_actiuni - 1][poz] = new_flower[:nr_actiuni - 1][poz] - sum
                sum = 0
            else:
                sum = sum - new_flower[:nr_actiuni - 1][poz]
                new_flower[:nr_actiuni - 1][poz] = 0
    ps = new_flower[:nr_actiuni - 1]
    y = ps.reshape([nr_actiuni - 1, 1])
    val, V = fobiectiv(Q, rmed, B, alpha, ro, Rp, y)
    new_flower[nr_actiuni - 1] = 1 / (1 + val)
    return new_flower



def FPA():
    Q, rmed, alpha, B, nr_actiuni = citeste_date("portofoliu3.txt")
    ro=10
    Rp=0.2
    dim=50
    switch_propability=0.4
    Lambda=1.5
    t = 0
    maxGeneration = 500
    pop = gen_pop(dim, nr_actiuni, Q, rmed, B, alpha, ro, Rp)
    g=pop[0]
    history = np.zeros([maxGeneration + 1, nr_actiuni])
    history[t] = g
    for i in range(dim):
        if pop[i][nr_actiuni-1]>g[nr_actiuni-1]:
            g=pop[i]
    while t<maxGeneration:
        for i in range(dim):
            random=np.random.uniform(0,1)
            if random<=switch_propability:
                L=draw_Levy_distribution_vector(nr_actiuni,Lambda)
                new_flower=global_polination(nr_actiuni, Q, rmed, B, alpha, ro, Rp,L,pop[i],g)
            else:
                E=draw_from_uniform_distribution(nr_actiuni)
                poz=np.random.randint(0,dim)
                poz1=np.random.randint(0,dim)
                while poz1==poz:
                    poz1 = np.random.randint(0, dim)
                new_flower=local_polination(nr_actiuni, Q, rmed, B, alpha, ro, Rp,E,pop[i],pop[poz],pop[poz1])
            if new_flower[nr_actiuni-1]>pop[i][nr_actiuni-1]:
                pop[i]=new_flower
        g = pop[0]
        for i in range(dim):
            if pop[i][nr_actiuni - 1] > g[nr_actiuni-1]:
                g = pop[i]
        if history[t][nr_actiuni-1]>=pop[0][nr_actiuni-1]:
            history[t+1]=history[t]
        else:
            history[t + 1] = g
        print(history[t][nr_actiuni-1])
        t+=1
    # print(history[len(history) - 1][nr_actiuni - 1])
    x = history[len(history) - 1][:nr_actiuni - 1]
    history_v = np.zeros(t)
    for i in range(t):
        history_v[i] = history[i][nr_actiuni - 1]
    print("sol")
    print(x)
    print("istoric_v")
    print(history_v)
    arata(x, nr_actiuni - 1, history_v, Q, rmed, B, alpha, ro, Rp)


FPA()



