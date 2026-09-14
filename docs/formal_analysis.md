# Formal status of KoRA

This document records what is proved, what is disproved, and what remains an
empirical hypothesis for the redesigned adapter. It does not claim that the
current vision notebook implements this operator.

For a fixed local feature vector x, LoRA gives `Mx` with `rank(M) <= R`.
The redesigned adapter gives `Mx + U((Px) * (Qx))`. Under a centrally
symmetric isotropic input distribution and squared population loss, linear
terms are odd and the multiplicative term is even, hence they are orthogonal.
If the teacher is `Tx + gamma*q(x)`, where q is exactly representable by the
interaction branch, the optimal risks are

`R_L = sum(j>R) sigma_j(T)^2 + gamma^2 E||q(x)||^2 + noise`

and

`R_K = sum(j>r) sigma_j(T)^2 + noise`.

Therefore KoRA wins in this local oracle setting iff the captured interaction
energy exceeds the linear singular-value energy sacrificed by using rank `r`
instead of `R`. This is a conditional approximation result, not a theorem
about full Transformer accuracy, optimization, or generalization.

Universal superiority is false: if the task is exactly representable by LoRA,
LoRA already has zero approximation error. Under a fixed parameter budget,
the interaction branch can also lose when the target is mostly linear.

The repository's former notebook is a pooled frozen-feature side network. Its
linear projection followed by token averaging collapses algebraically to a
projection of the pooled feature. Hooks record activations and do not modify
the backbone, so intermediate backbone CKA cannot be attributed to changed
backbone representations by that implementation.

The next proof obligations are budget-matched finite-sample risk, convergence
of the factorized non-convex optimization, and a separation that survives the
actual frozen attention computation. See `MultiplicativeKoRA` for the precise
operator and its tests.
