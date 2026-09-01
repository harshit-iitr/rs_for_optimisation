"""Full re-analysis under the characterization framing.

Organized as the paper would be: one section per proposition, then the battery,
then the aside. Reads only completed Round-5 runs; the archive is unreachable.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy import stats
from analysis.common import EXP, load_arm, per_seed, paired, fmt_paired, set_preliminary, window

set_preliminary(True)
SC = os.path.join(EXP, 'permuted_mnist/stiffness_curve')
LR = os.path.join(EXP, 'permuted_mnist/lr_frontier')
IC = os.path.join(EXP, 'permuted_mnist/isotropic_control')

LAM = ['lambda_0.0000','lambda_0.0001','lambda_0.0030','lambda_0.0060','lambda_0.0100',
       'lambda_0.1000','lambda_1.0000','lambda_10.0000']
LIM = ['limit_tangential','limit_ste']
ALLM = ['prev_only_acc','test_acc','phi_rad_tilde','radial_excess','radius_mean',
        'radius_std','weight_norm','eff_rank','stable_rank','dead_frac','dormant_frac',
        'readiness','update_norm','grad_norm','g_rad_norm','g_norm_task',
        'drift_abs','drift_rad_abs','drift_tan_abs','drift_rel','subspace_overlap',
        'drift_abs_ref','drift_rad_abs_ref','drift_tan_abs_ref','subspace_overlap_ref']

def L(root, arm, seeds=(1,2,3)):
    try: return load_arm(os.path.join(root, arm), list(seeds))
    except Exception: return None

def tab(rows, cols, fmt="{:9.4g}", w=9):
    hdr = "{:18s}".format('arm') + "".join(("{:>%ds}"%w).format(c[:w]) for c in cols)
    print(hdr); print("-"*len(hdr))
    for name, s in rows:
        print("{:18s}".format(name) + "".join(fmt.format(s[c]) for c in cols))

def sec(t):
    print("\n" + "="*86); print(t); print("="*86)

cache = {}
def get(root, arm):
    k=(root,arm)
    if k not in cache: cache[k]=L(root,arm)
    return cache[k]

def stat(root, arm, layer, metrics=ALLM):
    d = get(root, arm)
    if d is None: return None
    s = per_seed(d, metrics, layer=layer)
    return s.mean(), s.std(ddof=1), len(s)

# ============================================================ P1
sec("P1  RADIAL-ANGULAR DECOMPOSITION  --  phi_rad_tilde, per layer")
print("null under isotropic random gradient = 1.0 for every layer\n")
rows=[]
for a in LAM+LIM:
    r = {}
    ok=True
    for l in range(3):
        st = stat(SC, a, l, ['phi_rad_tilde','prev_only_acc','g_rad_norm','g_norm_task'])
        if st is None: ok=False; break
        r[f'phi_L{l}'] = st[0]['phi_rad_tilde']
        if l==0: r['prev_only']=st[0]['prev_only_acc']; r['n']=st[2]
    if ok: rows.append((a.replace('lambda_','lam=').replace('limit_','HARD '), r))
tab(rows, ['n','prev_only','phi_L0','phi_L1','phi_L2'])

soft=[(n,s) for n,s in rows if n.startswith('lam')]
y=np.array([s['prev_only'] for _,s in soft])
print("\n  Spearman(phi_rad, retention) over the 8 soft arms:")
for l in range(3):
    x=np.array([s[f'phi_L{l}'] for _,s in soft])
    rho,p=stats.spearmanr(x,y); print(f"    layer {l}: rho={rho:+.3f}  p={p:.4f}")
print("\n  The two HARD arms sit outside this relation entirely:")
for n,s in rows:
    if n.startswith('HARD'):
        print(f"    {n:18s} phi_L0={s['phi_L0']:.4f}  retention={s['prev_only']:.4f}")

# ============================================================ P2
sec("P2  WHAT THE PENALTY DOES TO WEIGHTS AND RANK (vs what lr does)")
print("If the penalty were an effective-lr reduction, the lr sweep should")
print("reproduce its weight/rank signature. Matched on radial_excess:\n")
rows=[]
for a in ['lambda_0.0000','lambda_0.0030','lambda_0.0060','lambda_0.0100','lambda_10.0000']:
    st=stat(SC,a,0)
    if st: rows.append(('PEN '+a.replace('lambda_',''), st[0]))
for a in sorted(os.listdir(LR)) if os.path.isdir(LR) else []:
    if not a.startswith('lr_'): continue
    st=stat(LR,a,0)
    if st: rows.append(('LR  '+a.replace('lr_',''), st[0]))
tab(rows, ['prev_only_acc','test_acc','radial_excess','radius_mean','radius_std',
           'weight_norm','eff_rank','stable_rank'])

# ============================================================ P3
sec("P3  THE RATCHET  --  is inflation cumulative across tasks?")
print("baseline vs penalty, layer 0, radius and weight norm by task index\n")
pts=[0,1,5,10,25,50,75,100,125,149]
print("{:16s}".format('arm')+"".join("{:>8s}".format(f't{t}') for t in pts))
for a in ['lambda_0.0000','lambda_0.0030','lambda_10.0000','limit_tangential']:
    d=get(SC,a)
    if d is None: continue
    for m in ['radius_mean','weight_norm']:
        g=d[d.layer==0].groupby('task')[m].mean()
        lbl=f"{a.replace('lambda_','lam').replace('limit_','H_')[:9]}:{m[:6]}"
        print("{:16s}".format(lbl)+"".join("{:8.2f}".format(g.get(t,np.nan)) for t in pts))

print("\n  cumulative vs consecutive drift, layer 0, final window:")
rows=[]
for a in LAM+LIM:
    st=stat(SC,a,0)
    if st: rows.append((a.replace('lambda_','lam=').replace('limit_','HARD '), st[0]))
tab(rows, ['prev_only_acc','drift_abs','drift_rad_abs','drift_tan_abs',
           'drift_abs_ref','drift_rad_abs_ref','drift_tan_abs_ref'])
soft=[(n,s) for n,s in rows if n.startswith('lam')]
y=np.array([s['prev_only_acc'] for _,s in soft])
print("\n  Spearman vs retention (8 soft arms):")
for c in ['drift_abs','drift_rad_abs','drift_tan_abs','drift_abs_ref',
          'drift_rad_abs_ref','drift_tan_abs_ref','subspace_overlap','subspace_overlap_ref']:
    x=np.array([s[c] for _,s in soft])
    if np.isnan(x).any(): continue
    rho,p=stats.spearmanr(x,y); print(f"    {c:22s} rho={rho:+.3f}  p={p:.4f}")

# ============================================================ battery
sec("THE BATTERY  --  every logged diagnostic across lambda, layer 0")
rows=[]
for a in LAM+LIM:
    st=stat(SC,a,0)
    if st: rows.append((a.replace('lambda_','lam=').replace('limit_','HARD '), st[0]))
tab(rows,['prev_only_acc','test_acc','radius_std','dead_frac','dormant_frac',
          'readiness','update_norm','grad_norm','g_rad_norm','g_norm_task'])

# ============================================================ S1 as battery test
sec("S1 CONTROLS UNDER THE BATTERY  (loose regime, clip=10)")
print("Do any of the magnitude-matched controls reproduce the penalty's SIGNATURE,")
print("not just its retention?\n")
rows=[]
for a in ['arm_baseline','arm_penalty_lam0.003','arm_isotropic_per_layer',
          'arm_isotropic_global','arm_iso_wnorm']:
    st=stat(os.path.join(IC,'loose'),a,0)
    if st: rows.append((a.replace('arm_','')[:17], st[0]))
tab(rows,['prev_only_acc','phi_rad_tilde','radius_mean','radius_std','weight_norm',
          'eff_rank','stable_rank','drift_abs_ref'])

# ============================================================ the aside
sec("THE ASIDE  --  where does the ratchet go under hard projection?")
print("weight norm at task 0 -> 149, layer 0:\n")
for a in ['lambda_0.0000','lambda_0.0030','lambda_10.0000','limit_tangential','limit_ste']:
    d=get(SC,a)
    if d is None: continue
    g=d[d.layer==0].groupby('task')['weight_norm'].mean()
    r=d[d.layer==0].groupby('task')['radius_mean'].mean()
    print(f"  {a:18s} weight {g.iloc[0]:6.2f} -> {g.iloc[-1]:6.2f}  ({g.iloc[-1]-g.iloc[0]:+6.2f})"
          f"   radius {r.iloc[0]:6.2f} -> {r.iloc[-1]:6.2f}  ({r.iloc[-1]-r.iloc[0]:+6.2f})")
