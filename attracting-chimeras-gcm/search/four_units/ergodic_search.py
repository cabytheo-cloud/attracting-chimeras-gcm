"""
Search for the four-unit example among six-piece Markov maps with the shape of the map f of the main example.
"""
# Template of the paper's 6-piece Markov map: f(0)=1, valley bottom (c,0), valley slopes 5/4 (x1=c/5, x2=9c/5, f(x1)=f(x2)=c),
# rising lap to peak (xp,1), falling lap to (x3,xp), steep last piece to (1,0). N=4, eps=1/4, U=P x P x Q x Q.
import numpy as np, itertools
from fractions import Fraction as Fr
e=0.25
C=np.array([[1-e/2,e/4,e/4],[e/2,1-3*e/4,e/4],[e/2,e/4,1-3*e/4]])
def smin(a,bs): return min(np.linalg.svd(C@np.diag([a,x,y]),compute_uv=False).min() for x in bs for y in bs)
def fixed_ok(c,xp,x3,bL,bR,P,Q):
    # affine laps: (slope, intercept, domain)
    laps={'vl':(-1.25,1.25*c,(P[0],c)),'vr':(1.25,-1.25*c,(c,P[1])),
          'pl':(bL,1-bL*xp,(Q[0],xp)),'pr':(-bR,1+bR*xp,(xp,Q[1]))}
    best=None
    for b in itertools.product(['vl','vr'],['pl','pr'],['pl','pr']):
        s=np.array([laps[k][0] for k in b]); t=np.array([laps[k][1] for k in b])
        A=np.eye(3)-C*s[None,:]; p=np.linalg.solve(A,C@t)
        sl=min(min(x-laps[k][2][0],laps[k][2][1]-x) for k,x in zip(b,p))
        offdiag=abs(p[1]-p[2])
        if sl>0 and (best is None or (offdiag>1e-9, sl)>(best[2]>1e-9,best[1])): best=(b,sl,offdiag)
    return best
res=[]
for c in np.arange(0.136,0.2,0.004):
    x1,x2=c/5,9*c/5
    blk_up=(7*c+1)/8
    if blk_up>=x2: continue
    for xp in np.arange(x2+0.2,0.9,0.01):
        bL=(1-c)/(xp-x2)
        for x3 in np.arange(xp+0.05,0.99,0.005):
            bR=(1-xp)/(x3-xp); 
            if bR<=4/3: continue
            last=xp/(1-x3)
            zup=(7+c)/8
            if zup>=x3: continue
            # choose q1 between zup and x3, q0 on rising lap; grid
            best=None
            for q1 in np.linspace(zup,x3,12)[1:-1]:
                for q0 in np.linspace(x2,xp,40)[1:-1]:
                    qmin=min(1-bL*(xp-q0),1-bR*(q1-xp))
                    m=min(qmin/8-x1, x2-blk_up, 7*qmin/8-q0, q1-zup)
                    if best is None or m>best[0]: best=(m,q0,q1)
            if best[0]<=0: continue
            s=smin(1.25,(bL,bR))
            if s<=1: continue
            fo=fixed_ok(c,xp,x3,bL,bR,(x1,x2),(best[1],best[2]))
            res.append((best[0],s,c,xp,x3,bL,bR,last,best[1],best[2],fo))
res.sort(key=lambda r:-min(r[0],(r[1]-1)/2))
print(len(res))
for r in res[:15]:
    print("margin %.4f smin %.4f c %.3f xp %.3f x3 %.3f bL %.3f bR %.3f last %.2f Q=[%.3f,%.3f]"%r[:10], r[10])
