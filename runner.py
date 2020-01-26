import matplotlib.pyplot as plt
from matplotlib import animation
from automata import Automata, AutomataOptions

should_animate = True

steps = 100

# Create automata with default options
a = Automata()

print(a)

# Run for n-steps
a.run(steps)

fig = plt.figure()

_fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.set_title("Entropy")
ax2.set_title("Seed")
ax3.set_title("Automata")
ax1.plot(a.scores)

im_seed = ax2.imshow(a.states[0])
im_seed.set_array(a.states[0])

i = 0
im_automata = ax3.imshow(a.states[-1], animated=True)

if should_animate == True:
    # Using animations - https://matplotlib.org/3.1.1/gallery/animation/simple_anim.html
    def update_figure(*args):
        global i
        if i < (steps - 1):
            i += 1
        else:
            i = 0
        im_automata.set_array(a.states[i])
        return (im_automata,)

    ani = animation.FuncAnimation(fig, update_figure, blit=True)

    # NOTE: Animation can be saved using:
    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # from matplotlib.animation import FFMpegWriter
    # writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)
else:
    im_automata.set_array(a.states[-1])

plt.show()
