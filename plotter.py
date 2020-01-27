import matplotlib.pyplot as plt
from matplotlib import animation


class Plotter:
    def __init__(self, automata):
        self.automata = automata

    def plot(self, should_animate=False):
        fig = plt.figure()

        _fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        _fig.tight_layout()

        ax1.set_title("Entropy")
        # ax1.axis("equal")
        ax1.set(
            xlim=(0, len(self.automata.scores)),
            ylim=(min(self.automata.scores), max(self.automata.scores)),
        )

        ax2.set_title("Seed")
        ax3.set_title("Automata")
        ax1.plot(self.automata.scores)

        im_seed = ax2.imshow(self.automata.states[0])
        im_seed.set_array(self.automata.states[0])

        self.i = 0
        im_automata = ax3.imshow(self.automata.states[-1], animated=True)

        steps = self.automata.steps_actual
        if should_animate == True:
            # Using animations - https://matplotlib.org/3.1.1/gallery/animation/simple_anim.html
            def update_figure(*args):
                i = self.i
                if i < (steps - 1):
                    i += 1
                else:
                    i = 0
                self.i = i
                im_automata.set_array(self.automata.states[i])
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
            im_automata.set_array(self.automata.states[-1])

        plt.show()
