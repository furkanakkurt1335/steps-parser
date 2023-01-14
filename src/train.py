#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

"""Main script for training a parser based on a configuration file."""

import argparse

from pathlib import Path

from init_config import ConfigParser
from parse_corpus import reset_file, parse_corpus, run_evaluation

import os, json
# import wandb

from smtp_gmail import send_start_email, send_res_email
from eval_indfeats import eval_indfeats

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
job_id = int(os.environ.get('SLURM_JOB_ID'))

def main(config, eval_mode="basic"):
    """Main function to initialize model, load data, and run training.

    Args:
        config: Experimental configuration.
        eval_mode: Method to use in post-training evaluation: "basic" for basic UD, "enhanced" for enhanced UD.
          Default: "basic".
    """
    model = config.init_model()

    data_loaders = config.init_data_loaders(model)

    trainer = config.init_trainer(model, data_loaders["train"], data_loaders["dev"])

    train_type = config['train_type']
    treebank = config['treebank']
    send_start_email(train_type, treebank, job_id)
    trainer.train()

    if "test" in config["data_loaders"]["paths"]:
        eval_results = evaluate_best_trained_model(trainer, config, eval_mode=eval_mode)
    if train_type in ['feats-only', 'upos_feats']:
        eval_results['IndFeats'] = eval_indfeats(config['data_loaders']['paths']['test'], os.path.join(THIS_DIR, 'tests-parsed/{ji}.conllu'.format(ji=job_id)))
    update_scores(train_type, treebank, eval_results)
    print_eval_results(train_type, eval_results)
    # log_wandb(train_type, eval_results)

    pth_l = [i for i in os.listdir(str(config._save_dir)) if i.endswith('.pth')]
    for pth_t in pth_l:
        pth_path = str(config._save_dir / pth_t)
        os.remove(pth_path)
        print('Removed: {}'.format(pth_path))
        os.system('python3 /clusterusers/furkan.akkurt@boun.edu.tr/eval-ud/gitlab-repo/trains/slurm/run-one.py')

    send_res_email(train_type, treebank, job_id, eval_results)


def print_eval_results(train_type, eval_results):
    ufeats = eval_results['UFeats'].f1
    lemmas = eval_results['Lemmas'].f1
    upos = eval_results['UPOS'].f1
    uas = eval_results['UAS'].f1
    las = eval_results['LAS'].f1

    if train_type in ['feats-only', 'upos_feats']:
        ind_feats = eval_results['IndFeats']
        res = f'UFeats: {100*ufeats:.2f}, IndFeats: {ind_feats:.2f}'
    elif train_type == 'lemma-only':
        res = f'Lemmas: {100*lemmas:.2f}'
    elif train_type == 'pos-only':
        res = f'UPOS: {100*upos:.2f}'
    elif train_type in ['dep-parsing', 'dep-parsing_upos', 'dep-parsing_feats', 'dep-parsing_upos_feats', 'dep-parsing_lemma']:
        res = f'UAS: {100*uas:.2f}, LAS: {100*las:.2f}'
    print('Eval results: {}.'.format(res))

def update_scores(train_type, treebank, eval_results):
    home = os.path.expanduser('~')
    scores_path = os.path.join(home, 'eval-ud/gitlab-repo/trains/scores/scores.json')
    with open(scores_path, 'r') as f:
        scores = json.load(f)
    if train_type in ['feats-only', 'upos_feats']:
        ufeats = eval_results['UFeats'].f1; ufeats = float(f'{100*ufeats:.2f}')
        scores[train_type][treebank]['UFeats'].append(ufeats)
        ind_feats = eval_results['IndFeats']
        scores[train_type][treebank]['IndFeats'].append(ind_feats)
    elif train_type == 'lemma-only':
        lemmas = eval_results['Lemmas'].f1; lemmas = float(f'{100*lemmas:.2f}')
        scores[train_type][treebank]['Lemmas'].append(lemma)
    elif train_type == 'pos-only':
        upos = eval_results['UPOS'].f1; upos = float(f'{100*upos:.2f}')
        scores[train_type][treebank]['UPOS'].append(upos)
    elif train_type in ['dep-parsing', 'dep-parsing_upos', 'dep-parsing_feats', 'dep-parsing_upos_feats', 'dep-parsing_lemma']:
        uas = eval_results['UAS'].f1; uas = float(f'{100*uas:.2f}')
        las = eval_results['LAS'].f1; las = float(f'{100*las:.2f}')
        scores[train_type][treebank]['UAS'].append(uas)
        scores[train_type][treebank]['LAS'].append(las)
    scores[train_type][treebank]['jobs'].append(job_id)
    with open(scores_path, 'w') as f:
        json.dump(scores, f)

# def log_wandb(train_type, eval_results):
#     if train_type in ['feats-only', 'upos_feats']:
#         wandb.log({'UFeats': eval_results['UFeats'].f1})
#     elif train_type == 'lemma-only':
#         wandb.log({'Lemmas': eval_results['Lemmas'].f1})
#     elif train_type == 'pos-only':
#         wandb.log({'UPOS': eval_results['UPOS'].f1})
#     elif train_type in ['dep-parsing', 'dep-parsing_upos', 'dep-parsing_feats', 'dep-parsing_upos_feats', 'dep-parsing_lemma']:
#         wandb.log({'UAS': eval_results['UAS'].f1, 'LAS': eval_results['LAS'].f1})

def evaluate_best_trained_model(trainer, config, eval_mode="basic"):
    """Evaluate the model with best validation performance on test data after training.

    Args:
        trainer: Trainer used for training the model.
        config: Model configuration (must contain path to test data).
        eval_mode: Method to use in evaluation: "basic" for basic UD, "enhanced" for enhanced UD. Default: "basic".
    """
    checkpoint_path = Path(trainer.checkpoint_dir) / "model_best.pth"
    trainer._resume_checkpoint(checkpoint_path)

    logger = config.logger

    logger.info("Evaluation on test set:")

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    tests_parsed_folder_path = os.path.join(THIS_DIR, 'tests-parsed')
    test_parsed_path = os.path.join(tests_parsed_folder_path, '{s_id}.conllu'.format(s_id=job_id))
    if not(os.path.exists(tests_parsed_folder_path)):
        os.mkdir(tests_parsed_folder_path)

    with open(config["data_loaders"]["paths"]["test"], "r") as gold_test_file, \
         open(test_parsed_path, "w") as output_file:
        parse_corpus(config, gold_test_file, output_file, parser=trainer.parser)
        output_file = reset_file(output_file, test_parsed_path)
        gold_test_file = reset_file(gold_test_file, config["data_loaders"]["paths"]["test"])
        test_evaluation = run_evaluation(gold_test_file, output_file, mode=eval_mode)

    if eval_mode == "basic":
        logger.log_final_metrics_basic(test_evaluation, suffix="_test")
    elif eval_mode == "enhanced":
        logger.log_final_metrics_enhanced(test_evaluation, suffix="_test")
    else:
        raise Exception(f"Unknown evaluation mode {eval_mode}")

    logger.log_artifact(test_parsed_path)

    return test_evaluation


def init_config_modification(raw_modifications):
    """Turn a "raw" config modification string into a dictionary of key-value pairs to replace."""
    modification = dict()
    for mod in raw_modifications:
        key, value = mod.split("=", 1)

        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value == "true":
                    value = True
                elif value == "false":
                    value = False

        modification[key] = value

    return modification


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Graph-based enhanced UD parser (training mode)')
    argparser.add_argument('config', type=str, help='config file path (required)')
    argparser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    argparser.add_argument('-s', '--save-dir', default=None, type=str, help='model save directory (config override)')
    argparser.add_argument('-m', '--modification', default=None, type=str, nargs='+', help='modifications to make to'
                                                                                           'specified configuration file'
                                                                                           '(config override)')
    argparser.add_argument('-e', '--eval', type=str, default="basic", help='Evaluation type (basic/enhanced).'
                                                                           'Default: basic')
    args = argparser.parse_args()

    modification = init_config_modification(args.modification) if args.modification is not None else dict()
    if args.save_dir is not None:
        modification["trainer.save_dir"] = args.save_dir

    # wandb.init(project="eval-ud")
    # config_name = os.path.splitext(os.path.basename(args.config))[0]
    # wandb.run.name = config_name

    config = ConfigParser.from_args(args, modification=modification)
    main(config, eval_mode=args.eval)
