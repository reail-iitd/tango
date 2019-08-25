import argparse

def initParser():
    """
    Input arguments for the simulation
    """
    parser = argparse.ArgumentParser('This will simulate a world describe in a json file.')
    parser.add_argument('--world', 
                            type=str, 
                            required=True,
                            help='The json file to visualize')
    parser.add_argument('--input',
                            type=str,
                            required=False,
                            default="jsons/input.json",
                            help='The json file of input high level actions')
    parser.add_argument('--timestep',
                            type=float,
                            required=False,
                            default=1.0,
                            help='How quickly to step through the visualization')
    return parser.parse_args()
 