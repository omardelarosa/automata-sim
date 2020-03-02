import numpy as np
from pypianoroll import Multitrack, Track
from matplotlib import pyplot as plt

# f = "./data/chpn_op10_e01.midi"
# f2 = "./data/r30_stretched.mid"
f = "./data/chopin_examples/logic_layered/track1_a.mid"
f2 = "./data/chopin_examples/logic_layered/track1_b.mid"

# Create a pianoroll matrix, where the first and second axes represent time
# and pitch, respectively, and assign a C major chord to the pianoroll
pianoroll = np.zeros((96, 128))
C_maj = [60, 64, 67, 72, 76, 79, 84]
pianoroll[0:95, C_maj] = 100

# Create a `pypianoroll.Track` instance
track = Track(pianoroll=pianoroll, program=0, is_drum=False, name="my awesome piano")

# Parse a MIDI file to a `pypianoroll.Multitrack` instance
c_mt = Multitrack(f)
r30_mt = Multitrack(f2)

# c_mt.merge_tracks([0])

# Plot the pianoroll
fig, ax = c_mt.tracks[0].plot(grid="off", preset="frame")
fig, ax = r30_mt.tracks[0].plot(grid="off", preset="frame")


plt.show()
