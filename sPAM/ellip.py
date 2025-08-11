

import jax
import jax.numpy as jnp
from sPAM.carlson import rf, rd

def F(phi, m):
    r"""JAX implementation of the incomplete elliptic integral of the first kind 

    .. math::

        \[F\left(\phi,k\right)=\int_{0}^{\phi}\frac{\,\mathrm{d}\theta}{\sqrt{1-m{%
\sin}^{2}\theta}}]

    Without latex, it is the integral from 0 to phi of the function 1/sqrt(1-m*sin^2(theta)).

     Args:
       phi: arraylike, real valued.
       m: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the first kind, :math:`F(\phi, m)`

     Notes:
       ``ellipfinc`` does not support complex-valued inputs.
       ``ellipfinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    c = 1.0 / jnp.sin(phi)**2
    return rf(c - 1, c - m, c)
    

def E(phi, m):
    r"""JAX implementation of the incomplete elliptic integral of the second kind 

    .. math::

        \[E\left(\phi,k\right)=\int_{0}^{\phi}\sqrt{1-m{\sin}^{2}\theta}\,\mathrm{d}%
\theta\\]

    Without latex, it is the integral from 0 to phi of the function sqrt(1-m*sin^2(theta)).

     Args:
       phi: arraylike, real valued.
       m: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the second kind, :math:`E(\phi, k)`

     Notes:
       ``ellipeinc`` does not support complex-valued inputs.
       ``ellipeinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    c = 1.0 / jnp.sin(phi)**2
    return rf(c - 1, c - m, c) - m * rd(c - 1, c - m, c) / 3.0
