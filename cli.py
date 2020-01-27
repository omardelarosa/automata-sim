import argparse

from automata import (
    Automata,
    AutomataOptions,
    DEFAULT_THRESHOLDS,
    DEFAULT_UNIVERSE_GRID_SIZE,
    DEFAULT_STEPS,
    DEFAULT_NEIGHBORHOOD_GRID_SIZE,
    ALIVE_T,
    DEAD_T,
)
import glob
from plotter import Plotter

# Default flags
SHOULD_ANIMATE = False
SHOULD_PLOT = False
SHOW_STATS = False
LOG_PARAMS = False

parser = argparse.ArgumentParser(description="Run a cellular automata simulation.")

parser.add_argument(
    "--thresholds",
    metavar="T",
    type=float,
    nargs=4,
    default=DEFAULT_THRESHOLDS,
    help="Thresholds for kernel function. (default: {})".format(DEFAULT_THRESHOLDS),
)

parser.add_argument(
    "--alive-threshold",
    metavar="A",
    nargs=1,
    type=float,
    default=ALIVE_T,
    help="Threshold of 'alive' state. (default: {})".format(ALIVE_T),
)

parser.add_argument(
    "--universe-size",
    metavar="U",
    type=int,
    nargs=2,
    default=DEFAULT_UNIVERSE_GRID_SIZE,
    help="Universe size (x,y) (default: {})".format(DEFAULT_UNIVERSE_GRID_SIZE),
)

parser.add_argument(
    "--neighborhood-size",
    metavar="N",
    type=int,
    nargs=2,
    default=DEFAULT_NEIGHBORHOOD_GRID_SIZE,
    help="Kernel neighborhood size (x,y) (default: {})".format(
        DEFAULT_NEIGHBORHOOD_GRID_SIZE
    ),
)

parser.add_argument(
    "--steps",
    metavar="S",
    type=int,
    nargs=1,
    default=DEFAULT_STEPS,
    help="Number of steps to run simulation (default: {})".format(DEFAULT_STEPS),
)

parser.add_argument(
    "--animate",
    action="store_true",
    default=False,
    help="Enable matplot lib animation. (default: {})".format(SHOULD_ANIMATE),
)

parser.add_argument(
    "--plot",
    action="store_true",
    default=False,
    help="Show plot. (default: {})".format(SHOULD_PLOT),
)

parser.add_argument(
    "--stats",
    action="store_true",
    default=False,
    help="Show stats after run. (default: {})".format(SHOW_STATS),
)

parser.add_argument(
    "--log-params",
    action="store_true",
    default=False,
    help="Show params of automata after run. (default: {})".format(LOG_PARAMS),
)

parser.add_argument(
    "--save-to",
    type=str,
    default="",
    help="Save automata results to a given location. (default: '{}')".format(""),
)

parser.add_argument(
    "--load-from",
    type=str,
    default="",
    help="Load automata from given location (accepts globs). (default: '{}')".format(
        ""
    ),
)

args = parser.parse_args()

should_animate = args.animate

steps = args.steps

universe_grid_size = tuple(args.universe_size)
neighborhood_grid_size = tuple(args.neighborhood_size)
thresholds = tuple(args.thresholds)
alive_t = args.alive_threshold

# Create automata with default options
opts = AutomataOptions(
    steps, universe_grid_size, neighborhood_grid_size, thresholds, alive_t, DEAD_T
)

a = Automata(opts)

if args.log_params:
    print("\nparams:\n{}".format(a.options.to_dict()))

if len(args.load_from):
    load_path = args.load_from
    files = glob.glob(load_path)
    files.sort()
    a.load(files)
    # Load from filesystem
else:
    # Run for n-steps
    a.run()

if args.stats:
    print("\nstats:\n{}".format(a.to_dict()))

if args.plot:
    # Plot result
    p = Plotter(a)
    p.plot(should_animate)

if len(args.save_to):
    a.save(args.save_to)
