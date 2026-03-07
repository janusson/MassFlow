from matchms import calculate_scores, Spectrum
from matchms.similarity import CosineGreedy
import numpy as np

s1 = Spectrum(mz=np.array([100.0, 200.0]), intensities=np.array([1.0, 1.0]))
s2 = Spectrum(mz=np.array([100.0, 200.0]), intensities=np.array([1.0, 1.0]))

scores = calculate_scores([s1], [s2], CosineGreedy())
arr = scores.scores.to_array()
print(arr.dtype)
print(arr)
