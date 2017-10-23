'''Propagate using Numba.'''
import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def inner(f, fp, nx, ny, nz, nxi, model_padded2_dt2, dt, sources,
          sources_x, sources_y, sources_z, num_sources, source_len,
          fd_coeff, stepidx):
    '''Take one forward step.'''

    for z in prange(8, nz - 8):
        for y in range(8, ny - 8):
            for x in range(8, nxi + 8):
                f_xx = 3 * fd_coeff[0] * f[z, y, x] + \
                        fd_coeff[1] * \
                        (f[z, y, x + 1] + \
                        f[z, y, x - 1] + \
                        f[z, y + 1, x] + \
                        f[z, y - 1, x] + \
                        f[z + 1, y, x] + \
                        f[z - 1, y, x]) + \
                        fd_coeff[2] * \
                        (f[z, y, x + 2] + \
                        f[z, y, x - 2] + \
                        f[z, y + 2, x] + \
                        f[z, y - 2, x] + \
                        f[z + 2, y, x] + \
                        f[z - 2, y, x]) + \
                        fd_coeff[3] * \
                        (f[z, y, x + 3] + \
                        f[z, y, x - 3] + \
                        f[z, y + 3, x] + \
                        f[z, y - 3, x] + \
                        f[z + 3, y, x] + \
                        f[z - 3, y, x]) + \
                        fd_coeff[4] * \
                        (f[z, y, x + 4] + \
                        f[z, y, x - 4] + \
                        f[z, y + 4, x] + \
                        f[z, y - 4, x] + \
                        f[z + 4, y, x] + \
                        f[z - 4, y, x]) + \
                        fd_coeff[5] * \
                        (f[z, y, x + 5] + \
                        f[z, y, x - 5] + \
                        f[z, y + 5, x] + \
                        f[z, y - 5, x] + \
                        f[z + 5, y, x] + \
                        f[z - 5, y, x]) + \
                        fd_coeff[6] * \
                        (f[z, y, x + 6] + \
                        f[z, y, x - 6] + \
                        f[z, y + 6, x] + \
                        f[z, y - 6, x] + \
                        f[z + 6, y, x] + \
                        f[z - 6, y, x]) + \
                        fd_coeff[7] * \
                        (f[z, y, x + 7] + \
                        f[z, y, x - 7] + \
                        f[z, y + 7, x] + \
                        f[z, y - 7, x] + \
                        f[z + 7, y, x] + \
                        f[z - 7, y, x]) + \
                        fd_coeff[8] * \
                        (f[z, y, x + 8] + \
                        f[z, y, x - 8] + \
                        f[z, y + 8, x] + \
                        f[z, y - 8, x] + \
                        f[z + 8, y, x] + \
                        f[z - 8, y, x])

                fp[z, y, x] = model_padded2_dt2[z, y, x] * \
                    f_xx + 2 * f[z, y, x] - fp[z, y, x]

    for i in range(num_sources):
        sx = sources_x[i] + 8
        sy = sources_y[i] + 8
        sz = sources_z[i] + 8
        fp[sz, sy, sx] += model_padded2_dt2[sz, sy, sx] * \
            sources[i, stepidx] * dt

@jit(nopython=True)
def step(f, fp, nx, ny, nz, nxi, model_padded2_dt2, dx, dt, sources,
         sources_x, sources_y, sources_z, num_sources, source_len, num_steps):
    '''Step forward by num_steps time steps.'''

    fd_coeff = np.array([-924708642.0, 538137600.0, -94174080.0,
                         22830080.0, -5350800.0, 1053696.0,
                         -156800.0, 15360.0, -735.0], np.float32)
    fd_coeff /= 302702400 * dx**2

    for stepidx in range(num_steps):
        inner(f, fp, nx, ny, nz, nxi, model_padded2_dt2, dt,
              sources, sources_x, sources_y, sources_z,
              num_sources, source_len, fd_coeff, stepidx)

        tmp = f
        f = fp
        fp = tmp
