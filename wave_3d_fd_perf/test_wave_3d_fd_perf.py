"""Test the propagators."""
import pytest
import numpy as np
from wave_3d_fd_perf.propagators import (VC1_O2_gcc, VC1_O3_gcc, VC1_Ofast_gcc,
                                         VC1_O2_unroll_gcc, VC1_O3_unroll_gcc,
                                         VC1_Ofast_unroll_gcc, VC2_Ofast_gcc,
                                         VC3_Ofast_gcc, VC4_O3_gcc,
                                         VC4_Ofast_gcc,
                                         VC5_Ofast_gcc, VC6_O3_gcc,
                                         VC6_Ofast_gcc,
                                         VC7_Ofast_gcc, VC7_O2_gcc,
                                         VC7_O2_unroll_gcc,
                                         VC8_Ofast_gcc, VC9_O3_gcc,
                                         VC9_Ofast_gcc,
                                         VC10_Ofast_gcc, VC11_Ofast_gcc,
                                         VC12_O3_gcc, VC12_Ofast_gcc,
                                         VF1_O2_gcc, VF1_O3_gcc, VF1_Ofast_gcc,
                                         VF2_Ofast_gcc, VF3_Ofast_gcc,
                                         VF4_O3_gcc, VF4_Ofast_gcc)

def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency."""
    t = np.arange(-peak_time, (length)*dt - peak_time, dt, dtype=np.float32)
    t = t[:length]
    y = ((1.0 - 2.0*(np.pi**2)*(freq**2)*(t**2))
         * np.exp(-(np.pi**2)*(freq**2)*(t**2)))
    return y

def green(x0, y0, z0, x1, y1, z1, dx, dt, T, v, f):
    """Use the 3D Green's function to determine the wavefield at a given
    location and time due to the given source.
    """
    r = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
    t = int((T - r/v)/dt)
    if (t >= 0) and (t < len(f)):
        return 1/(4*np.pi*r) * f[t] * dt * dx**3
    else:
        return 0.0

@pytest.fixture
def model_one(N=50, calc_expected=True):
    """Create a constant model, and the expected wavefield."""
    model = np.ones([N, N, N], dtype=np.float32) * 1500
    dx = 5
    dt = 0.001
    sx = int(N/2)
    sy = sx
    sz = sx
    # time is chosen to avoid reflections from boundaries
    T = (sx * dx / 1500)
    nsteps = np.ceil(T/dt).astype(np.int)
    source = ricker(25, nsteps, dt, 0.05)

    # direct wave
    if calc_expected:
        expected = np.array([green(x*dx, y*dx, z*dx, sx*dx, sy*dx, sz*dx,
                                   dx, dt,
                                   (nsteps)*dt, 1500,
                                   source) \
                             for x in range(N) \
                             for y in range(N) \
                             for z in range(N)])
        expected = expected.reshape([N, N, N])
    else:
        expected = []
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx]),
            'sy': np.array([sy]), 'sz': np.array([sz]),
            'expected': expected}


@pytest.fixture
def model_two(N=25, dt=0.0001):
    """Create a random model and compare with VPy1 implementation."""
    np.random.seed(0)
    model = np.random.random([N, N, N]).astype(np.float32) * 3000 + 1500
    dx = 5
    nsteps = np.ceil(0.2/dt).astype(np.int)
    num_sources = 10
    sources_x = np.zeros(num_sources, dtype=np.int)
    sources_y = np.zeros(num_sources, dtype=np.int)
    sources_z = np.zeros(num_sources, dtype=np.int)
    sources = np.zeros([num_sources, nsteps], dtype=np.float32)
    for sourceIdx in range(num_sources):
        sources_x[sourceIdx] = np.random.randint(N)
        sources_y[sourceIdx] = np.random.randint(N)
        sources_z[sourceIdx] = np.random.randint(N)
        peak_time = np.round((0.05 + np.random.rand() * 0.05) / dt) * dt
        sources[sourceIdx, :] = ricker(25, nsteps, dt, peak_time)
    v = VC1_O2_gcc(model, dx, dt)
    expected = v.step(nsteps, sources, sources_x, sources_y, sources_z)
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': sources, 'sx': sources_x, 'sy': sources_y,
            'sz': sources_z, 'expected': expected}


@pytest.fixture
def versions():
    """Return a list of implementations."""
    return [VC1_O2_gcc, VC1_O3_gcc, VC1_Ofast_gcc,
            VC1_O2_unroll_gcc, VC1_O3_unroll_gcc,
            VC1_Ofast_unroll_gcc, VC2_Ofast_gcc,
            VC3_Ofast_gcc, VC4_O3_gcc,
            VC4_Ofast_gcc,
            VC5_Ofast_gcc, VC6_O3_gcc,
            VC6_Ofast_gcc,
            VC7_Ofast_gcc, VC7_O2_gcc,
            VC7_O2_unroll_gcc,
            VC8_Ofast_gcc, VC9_O3_gcc,
            VC9_Ofast_gcc,
            VC10_Ofast_gcc, VC11_Ofast_gcc,
            VC12_O3_gcc, VC12_Ofast_gcc,
            VF1_O2_gcc, VF1_O3_gcc, VF1_Ofast_gcc,
            VF2_Ofast_gcc, VF3_Ofast_gcc,
            VF4_O3_gcc, VF4_Ofast_gcc]


def test_one_reflector(model_one, versions):
    """Verify that the numeric and analytic wavefields are similar."""

    for v in versions:
        _test_version(v, model_one, atol=1e-4)


def test_allclose(model_two, versions):
    """Verify that all implementations produce similar results."""

    for v in versions[1:]:
        print(v.__name__)
        _test_version(v, model_two, atol=5e-8)
        print(v.__name__, 'align 256')
        _test_version(v, model_two, atol=5e-8, align=256)


def _test_version(version, model, atol, align=None):
    """Run the test for one implementation."""
    v = version(model['model'], model['dx'], model['dt'], align=align)
    y = v.step(model['nsteps'], model['sources'], model['sx'], model['sy'],
               model['sz'])
    assert np.allclose(y, model['expected'], atol=atol)
