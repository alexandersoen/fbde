import math
import json
import time
import torch
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

from mollifiers.utils import discrete_result_statistics
from mollifiers.dataset import compas_dataset, adult_dataset, dutchlarge_dataset, german_dataset, dutch_dataset, cross_validate_dataset
from mollifiers.booster import DiscreteBoostedDensityEstimator, DiscreteEmpiricalDensityEstimator
from mollifiers.leverage import ExactLeverageSchedule, RelativeLeverageSchedule
from mollifiers.hypothesis import SKLearnDTHypothesisClass


def main() -> None:
    parser = argparse.ArgumentParser( description='Fair Mollifiers for Discrete Features Example' )

    # Constants
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Data selecting
    parser.add_argument('--dataset', type=str, default='compas', metavar='D',
                        help='selected dataset ["compas", "adult", "german", "dutch"] (default: "compas")')
    parser.add_argument('--sattr', nargs='+', type=str, default=['sex'],
                        help='specify column name as a sensitive attribute (default: ["sex"])')

    # Data splitting
    parser.add_argument('--cvsplits', type=int, default=5,
                        help='number of cross validation splits (default: 5)')

    # Evaluation
    parser.add_argument('--kmeans-clusters', type=int, default=4, metavar='K',
                        help='number of kmeans cluster fitted for evaluation (default: 4)')

    # Files
    parser.add_argument('--save-folder', type=str, default='.',
                        help='save folder (default: ".")')
    
    args = parser.parse_args()
    print('\n#########')
    print(args)

    print( f'using random seed {args.seed}' )
    np.random.seed( args.seed )
    torch.manual_seed( args.seed )

    save_path = Path(args.save_folder, 'data.json')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'saving results to {save_path}')

    # Get selected dataset
    if args.dataset == 'compas':
        dataset = compas_dataset(args.sattr)
    elif args.dataset == 'adult':
        dataset = adult_dataset(args.sattr)
    elif args.dataset == 'german':
        dataset = german_dataset(args.sattr)
    elif args.dataset == 'dutch':
        dataset = dutchlarge_dataset(args.sattr)
    else:
        raise ValueError(f'Unknown Dataset String: {args.dataset}')

    # Set up cross validation loop
    stats = []
    dataset_cv = cross_validate_dataset(dataset, n_splits=args.cvsplits, random_state=args.seed)
    for cv_i, (cur_train_dataset, cur_test_dataset) in tqdm(enumerate(dataset_cv), total=5, desc='fold prog', position=0):
        booster = DiscreteEmpiricalDensityEstimator(
                cur_train_dataset.data, cur_train_dataset.info)
        booster.init()
        
        boost_stats = [ discrete_result_statistics(booster, cur_train_dataset, cur_test_dataset, 0, args.kmeans_clusters, args.seed) ]
        boost_time = 0
        
        stats.append({
            'cv_iter': cv_i,
            'boost_stats': boost_stats,
            'time': boost_time,
        })
        
        # Save
        with save_path.open(mode='w') as f:
            json.dump(stats, f, indent=4, separators=(',', ':'))

    print('#########\n')

if __name__ == '__main__':
    main()
