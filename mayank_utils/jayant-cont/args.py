from argparse import ArgumentParser
def make_args():
	parser = ArgumentParser()
	parser.add_argument("--directions", type=int, default=36,
	    help="num directions to consider")
	parser.add_argument("--step", type=int, default=50,
	    help="num directions to consider")
	parser.add_argument("--thresh", type=float, default=0,
	    help="min threshold to move")
	parser.add_argument("--year", type=int, default=None,
	    help="year to use")
	parser.add_argument("--file", type=str, default=None,
	    help="file for initialisation")
	parser.add_argument('--random_eps', dest='random_eps', action='store_true',
                        help='step is not fixed')

	# parser.set_defaults(random_eps = False)
	parser.set_defaults(random_eps = True)

	args = parser.parse_args()
	return args
