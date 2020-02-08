import numpy as np
from pypianoroll import Multitrack, Track
from matplotlib import pyplot as plt
from oned import run

# Create a pianoroll matrix, where the first and second axes represent time
# and pitch, respectively, and assign a C major chord to the pianoroll

steps = 96

beat_duration = 16

pianoroll = np.zeros((steps * beat_duration, 128))

states = run(steps)

C_maj_scale = np.array([60, 62, 64, 65, 67, 69, 71])

for t in range(steps):
    state = states[t]
    beat = C_maj_scale * np.array(state)
    beat_list = [int(x) for x in beat.tolist()]
    print("{} -> {}".format(state, beat_list))
    beat_idx = t * beat_duration
    pianoroll[beat_idx, beat_list] = 100

# Create a `pypianoroll.Track` instance
track = Track(pianoroll=pianoroll, program=0, is_drum=False, name="my awesome piano")

mt = Multitrack(tracks=[track])

mt.write("./test.mid")

# # Plot the pianoroll
# fig, ax = track.plot()
# plt.show()
