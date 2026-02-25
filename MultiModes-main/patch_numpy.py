import numpy as np
import scipy as sc

if not hasattr(np, 'math'):
    np.math = np.emath

if not hasattr(np.math, 'pi'):
    np.math.pi = np.pi