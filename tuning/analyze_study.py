"""
Analyze a saved Optuna study.

Usage:
    python tuning/analyze_study.py --model MSPCIFormer
    python tuning/analyze_study.py --model NBeats --nbeats_type generic
"""

import pickle
import argparse
import os
import optuna


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--nbeats_type', type=str, default='interpretable')
    args = parser.parse_args()

    model_key = f'NBeats_{args.nbeats_type}' if args.model == 'NBeats' else args.model

    study_path = f'./tuning/results/{model_key}_study.pkl'
    if not os.path.exists(study_path):
        print(f'No study found at {study_path}. Run tune_hyperparams.py first.')
        return

    with open(study_path, 'rb') as f:
        study = pickle.load(f)

    print(f'\n===== Study summary: {model_key} =====')
    print(f'Number of trials:      {len(study.trials)}')
    print(f'Number completed:      {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}')
    print(f'Number pruned:         {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}')
    print(f'Best validation loss:  {study.best_value:.7f}')
    print('Best params:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')

    # Parameter importances
    try:
        importances = optuna.importance.get_param_importances(study)
        print('\nParameter importances:')
        for k, v in importances.items():
            print(f'  {k}: {v:.4f}')
    except Exception as e:
        print(f'Could not compute importances: {e}')

    # Top 5 trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    print('\nTop 5 trials:')
    for t in completed[:5]:
        print(f'  Trial {t.number:3d} | vali_loss={t.value:.7f} | {t.params}')


if __name__ == '__main__':
    main()
