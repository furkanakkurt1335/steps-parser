import argparse, os, re, json

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

gold_sent_pattern = r'# sent_id = (.*?)\n# text = (.*?)\n(.*?)\n\n'
pred_sent_pattern = r'(.*?)\n\n'

def eval_indfeats(gold_filepath, pred_filepath):
    with open(gold_filepath, 'r', encoding='utf-8') as f:
        gold_tb = f.read()
    with open(pred_filepath, 'r', encoding='utf-8') as f:
        pred_tb = f.read()
    gold_sents = re.findall(gold_sent_pattern, gold_tb, re.DOTALL)
    pred_sents = re.findall(pred_sent_pattern, pred_tb, re.DOTALL)

    pred_correct = 0
    feat_count = 0
    mispred_feat_d = dict()
    feat_count_d = dict()
    for i in range(len(gold_sents)):
        g_sent_id, g_text, g_lines_str = gold_sents[i]
        p_lines_str = pred_sents[i]

        g_lines = g_lines_str.split('\n')
        p_lines = p_lines_str.split('\n')
        for j in range(len(g_lines)):
            g_feats = g_lines[j].split('\t')[5].split('|')
            p_feats = p_lines[j].split('\t')[5].split('|')
            g_feat_d = dict()
            p_feat_d = dict()
            for g_feat_t in g_feats:
                if g_feat_t == '_':
                    continue
                g_tag_t, g_val_t = g_feat_t.split('=')
                g_feat_d[g_tag_t] = g_val_t
            for p_feat_t in p_feats:
                if p_feat_t == '_':
                    continue
                p_tag_t, p_val_t = p_feat_t.split('=')
                p_feat_d[p_tag_t] = p_val_t
            all_feats_s = set(list(g_feat_d.keys()) + list(p_feat_d.keys()))

            feat_count += len(all_feats_s)
            for feat_t in all_feats_s:
                if feat_t not in feat_count_d.keys():
                    feat_count_d[feat_t] = 0
                feat_count_d[feat_t] += 1
                if feat_t not in mispred_feat_d.keys():
                    mispred_feat_d[feat_t] = {'all': 0, 'nmatch': 0, 'gnexist': 0, 'pnexist': 0}
                if feat_t in g_feat_d.keys() and feat_t in p_feat_d.keys() and g_feat_d[feat_t] == p_feat_d[feat_t]:
                    pred_correct += 1
                else:
                    if feat_t not in p_feat_d.keys():
                        mispred_feat_d[feat_t]['pnexist'] += 1
                    elif feat_t not in g_feat_d.keys():
                        mispred_feat_d[feat_t]['gnexist'] += 1
                    elif g_feat_d[feat_t] != p_feat_d[feat_t]:
                        mispred_feat_d[feat_t]['nmatch'] += 1
                    mispred_feat_d[feat_t]['all'] += 1
    f = (pred_correct / feat_count)*100
    return float(f'{f:.2f}')