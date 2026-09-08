# Bid privacy for the TPE (mentor note)

**To:** Satendra Kumar
**From:** Nakshatra Kanchan
**Re:** Can the third-party entity run OPT2 without seeing bid amounts?

This note is research only. Paper 1 stays a **cleartext TPE**. There is no crypto in the code.

---

## 1. What the TPE already hides — and what it still sees

The double auction was designed so the TPE never sees the **utility functions** \(S_w\) (HSP) and \(T_k\) (BS). Agents bid scalars \(\varrho_{wk}\) and \(\zeta_{kw}\). The TPE solves OPT2 on those scalars and announces \((d,r)\) and payments.

That is *not* bid privacy. The TPE **does** see the bid amounts. The question is whether we can keep the same allocation while hiding those scalars from the TPE.

---

## 2. Why proxy re-encryption (PRE) does not solve this

PRE (AFGH / BBS / later variants) is **access delegation**: Alice encrypts to herself; a proxy, given a re-encryption key, translates the ciphertext so Bob can decrypt. The proxy does not learn the plaintext, but it also **cannot add, multiply, or take logs**.

OPT2 needs arithmetic on the bids:

\[
\tilde d_{wk}=\frac{\varrho_{wk}}{\tilde\mu_{wk}},\qquad
\tilde r_{kw}=\frac{\tilde\mu_{wk}-\tilde\lambda_k}{\zeta_{kw}}.
\]

A PRE ciphertext cannot be used as a number in that loop. One spectrum-auction paper that mentions PRE uses it only as a **verification / forwarding** helper; the actual market math is done under homomorphic encryption. Treating PRE as the privacy primitive for this auction is a category error.

Encrypt-in-transit, then decrypt at the TPE, also fails the goal: the TPE then sees bids in the clear again.

---

## 3. Hard leakage in *this* allocation (even if you hide the bid vector)

Suppose the TPE is shown only the public outcomes \((d,r,\mu,\lambda)\) and is asked not to be told \(\varrho,\zeta\). For the truthful OPT2 rule those outcomes **invert**:

\[
\varrho_{wk}=d_{wk}\,\mu_{wk},\qquad
\zeta_{kw}=\frac{\mu_{wk}-\lambda_k}{r_{kw}}.
\]

Worse, the HSP payment is already the bid sum: \(\Gamma_w=\sum_k\varrho_{wk}\). Publishing payments publishes \(\sum\varrho\).

So “run OPT2 on ciphertexts, then decrypt only \(d,r,\mu,\lambda\)” does **not** hide bids. Any later scheme must either

- keep duals and payments encrypted as well (and prove correctness), or
- accept that a TPE who sees the KKT point can reconstruct bids.

---

## 4. What actually works (SOTA ranking)

Ranked by whether they can evaluate OPT2 **without a single party seeing every bid**.

| Rank | Primitive | What it can do here | Cost / assumption |
| --- | --- | --- | --- |
| 1 | Two non-colluding HE servers (PS-TRUST, PROST, ARMOR-style) | Additive / packed HE on bids; servers jointly run the dual loop; neither sees plaintext bids if they do not collude | Two honest-but-curious servers; communication per iteration |
| 2 | Threshold FHE / MPC (FACT 2024, PANDA, encrypted ADMM) | Full OPT2 / ADMM under shared keys; reconstructions only of the public allocation | Heavier crypto; need a threshold committee |
| 3 | TEE as the TPE (Trustee-style) | Cleartext OPT2 inside an enclave; bids enter sealed, only \((d,r)\) leave | Practical, but trust the hardware / side-channel story |
| — | PRE | Delegation of *who may decrypt*, not evaluation of OPT2 | Does not meet the goal |
| — | Commit–reveal | Stops bid sniping; the TPE still sees bids at reveal | Wrong threat model |
| — | Differential privacy on bids | Hides an individual; breaks incentive compatibility / KKT match | Wrong threat model |

Recommended later primitive (pick one, not both in paper 1):

1. **Two-server HE** if the threat model is “the TPE must not learn \(\varrho,\zeta\).”
2. **TEE TPE** if the goal is an implementable prototype on the same OPT2 loop.

Paper 1 does neither.

---

## 5. What paper 1 should say

The TPE of this paper sees cleartext bids. That is the same leakage as the original Kumar–Kumar protocol. Hiding bid scalars is **future work**; PRE is not a candidate.
