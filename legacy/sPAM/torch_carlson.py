# algorithms from Carlson 1994 (https://arxiv.org/pdf/math/9409227.pdf)

# import jax
# from jax import numpy as jnp
# from jax import config
# config.update("jax_enable_x64", True)

import torch

# relative error will be "less in magnitude than r" 
r = 1.0e-15

def rf(x, y, z):

    r"""JAX implementation of Carlson's :math:`R_\mathrm{F}`

    Computed using the algorithm in Carlson, 1994: https://arxiv.org/pdf/math/9409227.pdf

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.

     Returns:
       The value of the integral :math:`R_\mathrm{F}`

     Notes:
       ``rf`` does not support complex-valued inputs.
       ``rf`` requires `jax.config.update("jax_enable_x64", True)`
    """
    
    xyz = torch.tensor([x, y, z])
    A0 = torch.sum(xyz) / 3.0
    v = torch.max(torch.abs(A0 - xyz))
    Q = (3 * r) ** (-1 / 6) * v

    cond = lambda s: s['f'] * Q > torch.abs(s['An'])

    def body(i, s):

        xyz = s['xyz']
        lam = (
            torch.sqrt(xyz[0]*xyz[1]) 
            + torch.sqrt(xyz[0]*xyz[2]) 
            + torch.sqrt(xyz[1]*xyz[2])
        )

        s['An'] = 0.25 * (s['An'] + lam)
        s['xyz'] = 0.25 * (s['xyz'] + lam)
        s['f'] = s['f'] * 0.25

        return s

    s = {'f': 1, 'An':A0, 'xyz':xyz}
    # s = jax.lax.while_loop(cond, body, s)
    # s = jax.lax.fori_loop(0, 10, body, s)
    for i in range(0, 10):
        s = body(i, s)

    x = (A0 - x) / s['An'] * s['f']
    y = (A0 - y) / s['An'] * s['f']
    z = -(x + y)
    E2 = x * y - z * z
    E3 = x * y * z

    return (
        1 
        - 0.1 * E2 
        + E3 / 14 
        + E2 * E2 / 24 
        - 3 * E2 * E3 / 44
    ) / torch.sqrt(s['An'])


def rd(x, y, z):
    r"""JAX implementation of Carlson's :math:`R_\mathrm{D}`

    Computed using the algorithm in Carlson, 1994: https://arxiv.org/pdf/math/9409227.pdf

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.

     Returns:
       The value of the integral :math:`R_\mathrm{D}`

     Notes:
       ``rd`` does not support complex-valued inputs.
       ``rd`` requires `jax.config.update("jax_enable_x64", True)`
    """

    xyz = torch.tensor([x, y, z])
    A0 = 0.2 * (x + y + 3 * z)
    v = torch.max(torch.abs(A0 - xyz))
    Q = (0.25 * r) ** (-1 / 6) * v

    cond = lambda s: s['f'] * Q > torch.abs(s['An'])

    def body(i, s):

        xyz = s['xyz']
        lam = (
            torch.sqrt(xyz[0]*xyz[1]) 
            + torch.sqrt(xyz[0]*xyz[2]) 
            + torch.sqrt(xyz[1]*xyz[2])
        )

        s['An'] = 0.25 * (s['An'] + lam)
        s['t'] = s['t'] + s['f'] / (torch.sqrt(xyz[2]) * (xyz[2] + lam))
        s['xyz'] = 0.25 * (xyz + lam)
        s['f'] = s['f'] * 0.25

        return s

    s = {'f': 1, 'An': A0, 'xyz': xyz, 't': 0}
    # s = jax.lax.while_loop(cond, body, s)
    # s = jax.lax.fori_loop(0, 10, body, s)
    for i in range(0, 10):
        s = body(i, s)

    x = (A0 - x) * s['f'] / s['An']
    y = (A0 - y) * s['f'] / s['An']
    z = -(x + y) / 3

    E2 = x * y - 6 * z * z
    E3 = (3 * x * y - 8 * z * z) * z
    E4 = 3 * (x * y - z * z) * z * z
    E5 = x * y * z**3

    return s['f'] * (
        1 
        - 3 * E2 / 14 
        + E3 / 6 
        + 9 * E2 **2 / 88 
        - 3 * E4 / 22 
        - 9 * E2 * E3 / 52 
        + 3 * E5 / 26
    ) * s['An']**-1.5 + 3 * s['t']