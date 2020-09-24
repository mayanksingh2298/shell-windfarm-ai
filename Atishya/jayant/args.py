from argparse import ArgumentParser
def make_args():
	parser = ArgumentParser()
	parser.add_argument("--directions", type=int, default=12,
	    help="num directions to consider")
	parser.add_argument("--step", type=int, default=50,
	    help="num directions to consider")
	parser.add_argument('--random_eps', dest='random_eps', action='store_true',
                        help='step is not fixed')
	parser.add_argument('--load', type=str, help='load initial turbine positions', default="")

	# parser.set_defaults(random_eps = False)
	parser.set_defaults(random_eps = True)

	args = parser.parse_args()
	return args
