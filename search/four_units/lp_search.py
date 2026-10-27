"""
First exploration for the four-unit example: for given slopes, a linear program maximizes the trapping margin of
P x P x Q x Q over the positions of the laps of the local map (scipy.optimize.linprog).
"""
# Search N=4: block of 2 in valley region P, two free units in peak region Q.
import numpy as np, itertools
from scipy.optimize import linprog

def Cmat(e):
    return np.array([[1-e/2, e/4, e/4],[e/2, 1-3*e/4, e/4],[e/2, e/4, 1-3*e/4]])

def smin(e,a,b1,b2):
    return np.linalg.svd(Cmat(e)@np.diag([a,b1,b2]),compute_uv=False).min()

def lp(e,a,bL,bR):
    # vars: cP,wP,vP,cQ,wL,wR,hQ,d
    # f(P)=[vP, vP+a wP], f(Q)=[hQ-bL wL, hQ] with bR wR <= bL wL
    A=[];b=[]
    def le(coef,rhs): A.append(coef); b.append(rhs)
    z=lambda **k:[k.get(n,0) for n in ['cP','wP','vP','cQ','wL','wR','hQ','d']]
    # bounds
    le(z(vP=-1),0)                       # vP>=0
    le(z(vP=1,wP=a),1)                   # vP+a wP<=1
    le(z(hQ=1),1)
    le(z(hQ=-1,wL=bL),0)                 # qmin>=0
    le(z(cP=-1,wP=1),0)                  # cP-wP>=0
    le(z(cQ=1,wR=1),1)                   # cQ+wR<=1
    le(z(cP=1,wP=1,cQ=-1,wL=1),-0.01)    # gap
    le(z(wR=bR,wL=-bL),0)
    # minP=vP, maxP=vP+a wP, minQ=hQ-bL wL, maxQ=hQ
    # x' = (1-e) f + e m, m in [(2minP+2minQ)/4,(2maxP+2maxQ)/4]
    # block lower: (1-e)vP + e(vP+hQ-bL wL)/2 >= cP-wP+d
    le(z(vP=-((1-e)+e/2),hQ=-e/2,wL=e*bL/2,cP=1,wP=-1,d=1),0)
    # block upper: (1-e)(vP+a wP)+e(vP+a wP+hQ)/2 <= cP+wP-d
    le(z(vP=(1-e)+e/2,wP=a*((1-e)+e/2),hQ=e/2,cP=-1,wP_=0,d=1) if False else
       [-1,a*((1-e)+e/2)-1,(1-e)+e/2,0,0,0,e/2,1],0)
    # Q lower: (1-e)(hQ-bL wL)+e(vP+hQ-bL wL)/2 >= cQ-wL+d
    le([0,0,-e/2,1,-1+((1-e)+e/2)*bL,0,-((1-e)+e/2),1],0)
    # Q upper: (1-e)hQ+e(vP+a wP+hQ)/2 <= cQ+wR-d
    le([0,e*a/2,e/2,-1,0,-1,(1-e)+e/2,1],0)
    c=np.zeros(8); c[7]=-1
    bounds=[(0,1)]*7+[(None,0.2)]
    r=linprog(c,A_ub=np.array(A),b_ub=np.array(b),bounds=bounds,method='highs')
    return r

res=[]
for e in [0.2,0.25,0.3,1/3,0.35,0.4]:
    lo,hi=1/(1-e/2),1/(1-e)
    for a in np.linspace(lo,hi,7)[1:-1]:
        for bL in [1.5,1.75,2,2.25,2.5,3]:
            for bR in [1.5,1.75,2,2.25,2.5,3]:
                s=min(smin(e,a,x,y) for x in (bL,bR) for y in (bL,bR))
                if s<=1.0: continue
                r=lp(e,a,bL,bR)
                if r.status==0 and -r.fun>0:
                    res.append((-r.fun,e,a,bL,bR,s,r.x))
res.sort(key=lambda t:-t[0])
print(len(res))
for t in res[:12]:
    print("margin %.4f eps %.3f a %.3f bL %.2f bR %.2f smin %.4f"%t[:6], np.round(t[6][:7],4))
