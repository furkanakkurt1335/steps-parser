import smtplib, ssl
import json, os

port = 465

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(THIS_DIR, 'gmail-credentials.json'), 'r') as f:
    credentials = json.load(f)
sender_email = credentials['sender_email']
sender_password = credentials['sender_password']
receiver_email = credentials['receiver_email']

def send_email(train_type, treebank, job_id, eval_results):
    context = ssl.create_default_context()
    ufeats = eval_results['UFeats'].f1
    lemmas = eval_results['Lemmas'].f1
    upos = eval_results['UPOS'].f1
    uas = eval_results['UAS'].f1
    las = eval_results['LAS'].f1

    if train_type == 'feats-only':
        res = f'UFeats: {100*ufeats:.2f}'
    elif train_type == 'lemma-only':
        res = f'Lemmas: {100*lemmas:.2f}'
    elif train_type == 'pos-only':
        res = f'UPOS: {100*upos:.2f}'
    elif train_type == 'dep-parsing':
        res = f'UAS: {100*uas:.2f}, LAS: {100*las:.2f}'
    elif train_type == 'dep-parsing_upos':
        res = f'UAS: {100*uas:.2f}, LAS: {100*las:.2f}'
    elif train_type == 'dep-parsing_feats':
        res = f'UAS: {100*uas:.2f}, LAS: {100*las:.2f}'
    elif train_type == 'dep-parsing_upos_feats':
        res = f'UAS: {100*uas:.2f}, LAS: {100*las:.2f}'
    elif train_type == 'upos_feats':
        res = f'UFeats: {100*ufeats:.2f}'

    message = '''\
Subject: Training done on cmpeinspurgpu!

Hi Dear Furkan,

Mailing from cmpeinspurgpu. Training of type {tt} in treebank {tb} with job ID {job_id} is done.

Results are {res}.

Best,
Furkan
    '''.format(tt=train_type, tb=treebank, job_id=job_id, res=res)

    with smtplib.SMTP_SSL('smtp.gmail.com', port, context=context) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, message)
