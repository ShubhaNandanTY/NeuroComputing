"""Components of the potential Phi = -J + beta*V - mu*E - lambda*R (proposal eq. 8).

task.py        J and the output error dJ/ds_out (core)
base.py        TermContext / TermOutput contract + local-autograd helper (core)
potential.py   composes the active terms into learning signals (core)
diversity.py   V (Deyvik)
metabolic.py   E_met (Joohyun)
barrier.py     E_bar + projection (Joohyun)
regularizer.py R (Joohyun)
"""
