"""
Evaluation of a few candidate local maps with the template of ergodic_search.py and simple rational nodes
(trapping margin, smallest singular value of the reduced map); the first one is the map f^ of Section 3.1.
"""
import numpy as np, itertools
from fractions import Fraction as Fr
e=Fr(1,4)
Cf=np.array([[7/8,1/16,1/16],[1/8,13/16,1/16],[1/8,1/16,13/16]])
def smin(a,bs): return min(np.linalg.svd(Cf@np.diag([a,x,y]),compute_uv=False).min() for x in bs for y in bs)
def evaluate(c,xp,x3,verbose=False):
    x1,x2=c/5,9*c/5; bL=(1-c)/(xp-x2); bR=(1-xp)/(x3-xp)
    blk_up=(7*c+1)/8; zup=(7+c)/8
    best=None
    for q1 in [zup+(x3-zup)*Fr(k,20) for k in range(1,20)]:
        for q0 in [x2+(xp-x2)*Fr(k,200) for k in range(1,200)]:
            qmin=min(1-bL*(xp-q0),1-bR*(q1-xp))
            m=min(qmin/8-x1, x2-blk_up, 7*qmin/8-q0, q1-zup)
            if best is None or m>best[0]: best=(m,q0,q1)
    s=smin(1.25,(float(bL),float(bR)))
    return dict(x1=x1,x2=x2,bL=bL,bR=bR,first=(1-c)/x1,last=xp/(1-x3),margin=best[0],q0=best[1],q1=best[2],smin=s)
for c,xp,x3 in [(Fr(1,6),Fr(4,5),Fr(23,25)),(Fr(1,6),Fr(4,5),Fr(32,35)),(Fr(1,6),Fr(4,5),Fr(9,10)),(Fr(1,6),Fr(4,5),Fr(11,12)),
                (Fr(3,16),Fr(4,5),Fr(11,12)),(Fr(1,6),Fr(4,5),Fr(13,14)),(Fr(1,6),Fr(4,5),Fr(27,29))]:
    r=evaluate(c,xp,x3)
    print("c",c,"xp",xp,"x3",x3,"bL",r['bL'],"bR",r['bR'],"first %.2f last %.2f"%(r['first'],r['last']),
          "margin %.4f smin %.4f Q=[%.4f,%.4f]"%(float(r['margin']),r['smin'],float(r['q0']),float(r['q1'])))
