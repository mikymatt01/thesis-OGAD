from enum import Enum
import argparse
import os

class Action(Enum):
    generate_dataset = 'generate_dataset'
    graphon_diffusion = 'graphon_diffusion'
    evaluate_generated_graphs = 'evaluate_generated_graphs'

    def __str__(self):
        return self.value
    
parser = argparse.ArgumentParser(description="Arguments for training and evaluating graphon learning models.")
parser.add_argument('--action', type=Action, default=None, help="Action", choices=list(Action))

args = parser.parse_args()
action = args.action

if __name__ == "__main__":
    if action == Action.generate_dataset:
        os.system('poetry run python sigl/Main.py --Adjs_dir ../data/train/graphs/CommunityGraphs.pickle  --Res 32 --output_dir ../data/train/graphons --test_name train')
        os.system('poetry run python sigl/Main.py --Adjs_dir ../data/test/graphs/CommunityGraphs.pickle  --Res 32 --output_dir ../data/test/graphons --test_name test')
    if action == Action.graphon_diffusion:
        os.system()
    elif action == Action.evaluate_generated_graphs:
        os.system('poetry run python ggme/main.py')
    else:
        raise Exception("no action found")