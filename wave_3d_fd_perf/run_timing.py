"""Measure the runtime of the propagators."""
from timeit import repeat
import numpy as np
import pandas as pd
from wave_3d_fd_perf.propagators import (VC1_O2_gcc, VC1_O3_gcc, VC1_Ofast_gcc,
                                         VC1_O2_unroll_gcc, VC1_O3_unroll_gcc,
                                         VC1_Ofast_unroll_gcc, VC2_Ofast_gcc,
                                         VF1_O2_gcc, VF1_O3_gcc, VF1_Ofast_gcc,
                                         VF2_Ofast_gcc)
from wave_3d_fd_perf.test_wave_3d_fd_perf import ricker

def run_timing_num_steps(num_repeat=10, num_steps=range(0, 110, 10),
                         model_size=200, versions=None, align=None):
    """Time implementations as num_steps varies."""

    if versions is None:
        versions = _versions()

    times = pd.DataFrame(columns=['version', 'num_steps', 'model_size', 'time'])

    for nsteps in num_steps:
        model = _make_model(model_size, nsteps)
        times = _time_versions(versions, model, num_repeat, times, align)

    return times


def run_timing_model_size(num_repeat=10, num_steps=10,
                          model_sizes=range(50, 250, 50), versions=None,
                          align=None):
    """Time implementations as model size varies."""

    if versions is None:
        versions = _versions()

    times = pd.DataFrame(columns=['version', 'num_steps', 'model_size', 'time'])

    for N in model_sizes:
        model = _make_model(N, num_steps)
        times = _time_versions(versions, model, num_repeat, times, align)

    return times


def _versions():
    """Return a list of versions to be timed."""
    return [{'class': VC1_O2_gcc, 'name': 'C v1 (gcc, -O2)'},
            {'class': VC1_O3_gcc, 'name': 'C v1 (gcc, -O3)'},
            {'class': VC1_Ofast_gcc, 'name': 'C v1 (gcc, -Ofast)'},
            {'class': VC1_O2_unroll_gcc, 'name': 'C v1 (gcc, -O2 unroll)'},
            {'class': VC1_O3_unroll_gcc, 'name': 'C v1 (gcc, -O3 unroll)'},
            {'class': VC1_Ofast_unroll_gcc, 'name': 'C v1 (gcc, -Ofast unroll)'},
            {'class': VC2_Ofast_gcc, 'name': 'C v2 (gcc, -Ofast)'},
            {'class': VF1_O2_gcc, 'name': 'F v1 (gcc, -O2)'},
            {'class': VF1_O3_gcc, 'name': 'F v1 (gcc, -O3)'},
            {'class': VF1_Ofast_gcc, 'name': 'F v1 (gcc, -Ofast)'},
            {'class': VF2_Ofast_gcc, 'name': 'F v2 (gcc, -Ofast)'}]


def _make_model(N, nsteps):
    """Create a model with a given number of elements and time steps."""
    model = np.random.random([N, N, N]).astype(np.float32) * 3000 + 1500
    dx = 5
    dt = 0.0001
    source = ricker(25, nsteps, dt, 0.05)
    sx = int(N/2)
    sy = sx
    sz = sx
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx]),
            'sy': np.array([sy]), 'sz': np.array([sz])}


def _time_versions(versions, model, num_repeat, dataframe, align=None):
    """Loop over versions and append the timing results to the dataframe."""
    num_steps = model['nsteps']
    model_size = len(model['model'])
    for v in versions:

        time = _time_version(v['class'], model, num_repeat, align)
        dataframe = dataframe.append({'version': v['name'],
                                      'num_steps': num_steps,
                                      'model_size': model_size,
                                      'time': time}, ignore_index=True)
    return dataframe


def _time_version(version, model, num_repeat, align=None):
    """Time a particular version."""
    v = version(model['model'], model['dx'], model['dt'], align)

    def closure():
        """Closure over variables so they can be used in repeat below."""
        v.step(model['nsteps'], model['sources'], model['sx'], model['sy'],
               model['sz'])

    return np.min(repeat(closure, number=1, repeat=num_repeat))

if __name__ == '__main__':
    print(run_timing_num_steps())
