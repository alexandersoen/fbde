import math
import json
import time
import torch
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

from mollifiers.dataset import cross_validate_dataset, minneapolis_dataset

from mollifiers.utils import *

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
    parser.add_argument('--hypothesis', type=str, default='nn1', metavar='H',
                        help='weak learner hypothesis class ["nn1", "nn2"] (default: "nn1")')
    parser.add_argument('--clip', type=float, default=math.log(2), metavar='H',
                        help='clip value for hypothesis (default: "log(2)")')
    parser.add_argument('--equal-rr', action='store_true',
                        help='set initial distribution to have equal RR (default: "False")')

    # Data selecting
    parser.add_argument('--dataset', type=str, default='compas', metavar='D',
                        help='selected dataset ["compas", "adult"] (default: "compas")')

    # Data splitting
    parser.add_argument('--test-split', type=float, default=0.2, metavar='T',
                        help='percentage of original data used for testing (default: 0.2)')

    # Evaluation
    parser.add_argument('--kmeans-clusters', type=int, default=4, metavar='K',
                        help='number of kmeans cluster fitted for evaluation (default: 4)')
    parser.add_argument('--wass-bins', type=int, default=20, metavar='W',
                        help='number of wasserstein bins (default: 20)')

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
    if args.dataset == 'minneapolis':
        dataset = minneapolis_dataset()
        args.sattr = 'race'
    else:
        raise ValueError(f'Unknown Dataset String: {args.dataset}')


    def data_result_statistics(in_samples, train_data, test_data, boost_i, kmeans_clusters, seed, n_bins=20):
        boost_samples = in_samples
        info = train_data.info

        post_clf = DecisionTreeClassifier(max_depth=32)
        post_clf.fit(boost_samples[:, info.unlabel_columns], boost_samples[:, info.label_columns].ravel())

        base_clf = DecisionTreeClassifier(max_depth=32)
        base_clf.fit(train_data.data[:, info.unlabel_columns], train_data.data[:, info.label_columns].ravel())

        post_kmeans = KMeans(n_clusters=kmeans_clusters, random_state=seed)
        post_kmeans.fit(normalize(boost_samples[:, info.feature_columns + info.label_columns], axis=0))

        clf_samples = copy.deepcopy(test_data.data)
        clf_samples[:, info.label_columns] = post_clf.predict_proba(test_data.data[:, info.unlabel_columns])[:, 1].reshape(-1, 1)

        clustering_dict = {}
        clustering_dict.update(sklearn_kmean_privilege_ratio(post_kmeans, test_data.data, info))
        clustering_dict.update(sklearn_kmean_statistical_rate(post_kmeans, test_data.data, info))
        clustering_dict['kmeans_dist'] = sklearn_kmean_mahalanobis(post_kmeans, test_data.data, info),

        return {
            'boost_iter': boost_i,
            'boosting': {
                'theta': None,
                'norm': None,
                'wl_train_acc': None,
                'wl_train_loss': None,
            },
            'data': {
                'rr': representation_rate(boost_samples, info),
                'sr': statistical_rate(boost_samples, info),
                'train_wass2': w2_eval(train_data.data, boost_samples, n_bins),
                'test_wass2': w2_eval(test_data.data, boost_samples, n_bins),
                'train_kl': kl_cts_eval(train_data.data, boost_samples, n_bins),
                'test_kl': kl_cts_eval(test_data.data, boost_samples, n_bins),
            },
            'prediction': {
                'clf_sr': statistical_rate(clf_samples, info),
                'clf_acc': sklearn_accuracy(post_clf, test_data.data, info),
                'clf_eo': sklearn_equal_opportunity(post_clf, test_data.data, info),
            },
            'clustering': clustering_dict,
        }


    # Set up cross validation loop
    stats = []
    dataset_cv = cross_validate_dataset(dataset, n_splits=5, random_state=args.seed)
    for cv_i, (cur_train_dataset, cur_test_dataset) in tqdm(enumerate(dataset_cv), total=5, desc='fold prog', position=0):

        boost_stats = [ data_result_statistics(cur_train_dataset.data, cur_train_dataset,
                                                     cur_test_dataset, 0,
                                                     args.kmeans_clusters,
                                                     args.seed,
                                                     n_bins = args.wass_bins) ]
        
            
        stats.append({
            'cv_iter': cv_i,
            'boost_stats': boost_stats,
            'time': 0,
        })
        
        # Save
        with save_path.open(mode='w') as f:
            json.dump(stats, f, indent=4, separators=(',', ':'))

    print('#########\n')

if __name__ == '__main__':
    main()
