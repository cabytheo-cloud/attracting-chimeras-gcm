"""
Numerics for the four-unit example quoted in Section 3.1.

  1. Fate of 2x10^5 initial conditions drawn uniformly in [0,1]^4 (paper: about half converge to one of the six
     chimeras of Theorem 3.6, two units synchronized in the valley and two free units at the peak; the others to
     chimeras with three units synchronized in the valley and one unit at the peak).
  2. On the attractor of the reduced map G^: the three Lyapunov exponents (paper: about 0.30, 0.30, 0.07), the TLE of
     the pair (log 15/16), the average of Proposition 4.3 for the free units (0), how often the free units come within
     1e-3 of each other (paper: about 2% of the time), and the occupied volume.
Needs numpy only. Full run: about 5 minutes; --quick: 2x10^4 initial conditions (under a minute).
"""
import sys
import numpy as np
rng=np.random.default_rng(7); e=0.25; N=4
X=np.array([0,1/30,1/6,3/10,4/5,23/25,1.0]); Y=np.array([1,1/6,0,1/6,1,4/5,0])
f=lambda x: np.interp(x,X,Y)
SL=np.array([-25,-1.25,1.25,5/3,-5/3,-10])
def fp(x): return SL[np.clip(np.searchsorted(X,x,side='right')-1,0,5)]
def F(x): fx=f(x); return (1-e)*fx+e*fx.mean(axis=-1,keepdims=True)
P=(1/30,3/10); Q=(17/25,23/25)
M=20000 if '--quick' in sys.argv else 200000; x=rng.random((M,N)); entered=np.full(M,-1)
def inbox(x):
    inP=(x>P[0])&(x<P[1]); inQ=(x>Q[0])&(x<Q[1])
    return (inP.sum(1)==2)&(inQ.sum(1)==2)
for t in range(6000):
    ib=inbox(x); entered[(entered<0)&ib]=t
    x=F(x)
iu=np.triu_indices(N,1); dd=np.abs(x[:,iu[0]]-x[:,iu[1]]); sync=dd<1e-10; ns=sync.sum(1)
inPv=((x>P[0])&(x<P[1])).sum(1)
print("final states from %d random ICs:"%M)
for k in sorted(set(ns.tolist())):
    sel=ns==k; vals,cnt=np.unique(inPv[sel],return_counts=True)
    print("  synced pairs=%d: %.4f   (units in P: %s)"%(k,sel.mean(),dict(zip(vals.tolist(),(cnt/sel.sum()).round(3).tolist()))))
ch=(ns==1)
print("2+2free chimera fraction %.4f; of these entered box U: %.4f; entry time median %d max %d"%(ch.mean(),(entered[ch]>=0).mean(),np.median(entered[ch]),entered[ch].max()))
# reduced 3D dynamics
C=np.array([[1-e/2,e/4,e/4],[e/2,1-3*e/4,e/4],[e/2,e/4,1-3*e/4]])
def G(v): return (C@f(v).T).T
K=4000; v=np.column_stack([rng.uniform(*P,K),rng.uniform(*Q,K),rng.uniform(*Q,K)])
for t in range(300): v=G(v)
Qm=np.tile(np.eye(3),(K,1,1)); S=np.zeros((K,3)); T=3000
dmin=np.full(K,np.inf); below=0; sec=np.zeros(K); tle=np.zeros(K); pts=[]
for t in range(T):
    y,z3,z4=v.T
    J=C[None]*np.stack([fp(y),fp(z3),fp(z4)],1)[:,None,:]
    Qm,R=np.linalg.qr(J@Qm); S+=np.log(np.abs(np.diagonal(R,axis1=1,axis2=2)))
    dz=np.abs(z3-z4); dmin=np.minimum(dmin,dz); below+=(dz<1e-3).sum()
    sec+=np.log(0.75*np.abs((f(z3)-f(z4))/(z3-z4))); tle+=np.log(0.75*np.abs(fp(y)))
    if t%5==0: pts.append(v.copy())
    v=G(v)
L=np.sort(S.mean(0)/T)[::-1]
print("Lyapunov exponents in the cluster space:",L.round(4)," log sigma =",round(np.log(1.0174617),4))
print("block TLE average",tle.mean()/T,"(log 15/16 =",np.log(15/16),")")
print("pair (3,4): mean log((3/4)|f[z3,z4]|) = %.2e ; min |z3-z4| = %.2e ; fraction of time |z3-z4|<1e-3: %.4f"%(sec.mean()/T,dmin.min(),below/(K*T)))
pts=np.concatenate(pts)
print("attractor ranges: y [%.4f,%.4f], z [%.4f,%.4f]"%(pts[:,0].min(),pts[:,0].max(),pts[:,1:].min(),pts[:,1:].max()))
for h in [1/100,1/200]:
    cells=np.unique(np.floor(pts/h).astype(np.int64),axis=0)
    print("occupied volume, grid %.4f: %.6f"%(h,len(cells)*h**3))
