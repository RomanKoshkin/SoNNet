import sys
import argparse
import yaml
from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dishbrain simulator")
    parser.add_argument("--config", default=None, help="Config file to use (see configs dir)")
    parser.add_argument("--niter", type=int, default=1000, help="Number of iterations of the simulation")
    parser.add_argument("--K", type=int, default=32, help="Number of harmonics in fourier space for spatial attention")
    # parser.add_argument("--wav2vec-model", type=str, default="xlsr_53_56k", help="Type of wav2vec2.0 model to use")
    # parser.add_argument("--reproducible", action='store_true', help="Seed everything")

    args = parser.parse_args()

    if len(sys.argv) > 1:
        get_config(args)
    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()