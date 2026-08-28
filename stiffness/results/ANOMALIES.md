# Anomalies and Negative Results

## S4.1: M1 Verification Failure
**Finding:** The theoretical prediction (T1) states that the steady-state radial excess $u^*$ under the soft Radial Suppression penalty should follow $u^* = \frac{g_{rad} \cdot d}{2\lambda}$. This predicts that on a log-log plot of steady-state radial excess vs $\lambda$, the slope should be $-1$. 

**Observation:** The empirical slope is significantly flatter than $-1$ and varies dramatically by layer depth.
- Layer 0: slope $\approx -0.05$ to $-0.08$
- Layer 1: slope $\approx -0.25$ to $-0.32$
- Layer 2: slope $\approx -0.40$ to $-0.57$

None of the layers fall within the expected $[-1.3, -0.7]$ boundary. The stiffness penalty operates entirely differently in practice than the theoretical abstraction suggests.

## S2: Baselines Head-to-Head Failure (Kill Condition)
**Finding:** The S2 baseline sweep compared RS against standard regularizers and continuous learning methods. The plan states:
> *"Kill condition S2: RS does not clearly beat standard L2 on Final Acc."*

**Observation:**
- **L2 ($\lambda=1e-3$):** Acc: $78.17\% \pm 0.21\%$
- **RS (Best $\lambda=1.0$):** Acc: $77.72\% \pm 0.18\%$

A paired t-test between RS and L2 ($1e-3$) yields $t = -3.665, p = 0.0215$. RS is statistically **worse** than a well-tuned standard weight decay.

**Conclusion:** Radial Suppression fails the empirical bar. It does not provide an advantage over simple L2 regularization. 

### Final Project Status: HALTED
Both the theoretical prediction (S4.1) and the empirical benchmark (S2) have explicitly failed their kill criteria. Following the rigorous methodology outlined in `plan.md`, development on the Radial Suppression algorithm should be halted, as the evidence shows it is neither theoretically grounded by T1 nor empirically superior to simple baselines. This is a legitimate negative result that refutes M1 and reshapes the paper's theoretical claims.
