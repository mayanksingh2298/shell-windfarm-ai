from argparse import ArgumentParser
def make_args():
	parser = ArgumentParser()
	parser.add_argument("--directions", type=int, default=12,
	    help="num directions to consider")
	parser.add_argument("--step", type=int, default=50,
	    help="num directions to consider")


	args = parser.parse_args()
	return args
