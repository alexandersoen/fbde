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
from mollifiers.booster import DiscreteBoostedDensityEstimator
from mollifiers.leverage import ExactLeverageSchedule, RelativeLeverageSchedule
from mollifiers.hypothesis import SKLearnDTHypothesisClass, SKLearnDTUncaliHypothesisClass


def main() -> None:
    parser = argparse.ArgumentParser( description='Fair Mollifiers for Discrete Features Example' )

    # Constants
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Boosting settings
    parser.add_argument('--boosting-steps', type=int, default=50, metavar='T',
                        help='specify number of boosting steps (default: 50)')
    parser.add_argument('--init-sr', type=float, default=1, metavar='R',
                        help='initial density representation rate (default: 1)')
    parser.add_argument('--sr', type=float, default=0.9, metavar='R',
                        help='designated representation rate (default: 0.9)')
    parser.add_argument('--leverage', type=str, default='exact', metavar='L',
                        help='specify leveraging function (default: "exact")')
    parser.add_argument('--max-depth', type=int, default=8, metavar='D',
                        help='specify decision tree maximum depth (default: 8)')
    parser.add_argument('--hypothesis', type=str, default='dt', metavar='H',
                        help='weak learner hypothesis class ["dt", "nn"] (default: "dt")')
    parser.add_argument('--clip', type=float, default=math.log(2), metavar='H',
                        help='weak learner hypothesis class ["dt", "nn"] (default: "dt")')
    parser.add_argument('--equal-rr', action='store_true',
                        help='set initial distribution to have equal RR (default: "False")')
    parser.add_argument('--mix', type=float, default=0,
                        help='mixing parameter for the initial distribution (default: 0)')

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
    parser.add_argument('--save-file', type=str, default='mollifier_res.json',
                        help='save file (default: "mollifier_res.json")')
    
    args = parser.parse_args()
    print('\n#########')
    print(args)

    print( f'using random seed {args.seed}' )
    np.random.seed( args.seed )
    torch.manual_seed( args.seed )

    save_path = Path(args.save_folder, args.save_file)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'saving results to {save_path}')

    hypothesis_path = Path(args.save_folder, 'hypothesis', args.save_file.rsplit('.', 1)[0])
    hypothesis_path.mkdir(parents=True, exist_ok=True)
    print(f'saving hypothsis to {hypothesis_path}')

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

    print(f'hypothesis max depth: {args.max_depth}')

    if args.hypothesis == 'dt':
        hypothesis_class = SKLearnDTHypothesisClass(args.clip, dataset.info, max_depth=args.max_depth)
    elif args.hypothesis == 'dtu':
        hypothesis_class = SKLearnDTUncaliHypothesisClass(args.clip, dataset.info, max_depth=args.max_depth)
        
    if args.leverage == 'exact':
        leverage_func = ExactLeverageSchedule(C=args.clip)
    elif args.leverage == 'relative':
        leverage_func = RelativeLeverageSchedule(C=args.clip)
    else:
        raise ValueError(f'Unknown Leverage String: {args.leverage}')

    # Set up cross validation loop
    stats = []
    dataset_cv = cross_validate_dataset(dataset, n_splits=args.cvsplits, random_state=args.seed)
    for cv_i, (cur_train_dataset, cur_test_dataset) in tqdm(enumerate(dataset_cv), total=5, desc='fold prog', position=0):
        tau = args.sr

        booster = DiscreteBoostedDensityEstimator(cur_train_dataset.data,
                                                  cur_train_dataset.info, tau,
                                                  hypothesis_class,
                                                  leverage_func, args.init_sr,
                                                  equal_rr = args.equal_rr,
                                                  mix=args.mix)
        booster.init()
        
        boost_stats = [ discrete_result_statistics(booster, cur_train_dataset, cur_test_dataset, 0, args.kmeans_clusters, args.seed) ]
        boost_time = 0
        
        for boost_iter in tqdm(range(args.boosting_steps), desc=' boost iter', position=1, leave=False):
            # Computing samples for stats logging

            ### Boosting Step ###
            cur_start_time = time.time()
            booster.step()
            cur_end_time = time.time()
            ### Boosting Step ###

            # Record stats
            boost_stats.append(discrete_result_statistics(booster, cur_train_dataset,
                                                          cur_test_dataset,
                                                          boost_iter+1,
                                                          args.kmeans_clusters,
                                                          args.seed))
            boost_time += cur_end_time - cur_start_time
            
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
