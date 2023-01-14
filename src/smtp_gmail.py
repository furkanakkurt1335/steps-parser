import smtplib, ssl
import json, os, subprocess
from email.message import EmailMessage

port = 465

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(THIS_DIR, 'gmail-credentials.json'), 'r') as f:
    credentials = json.load(f)
sender_email = credentials['sender_email']
sender_password = credentials['sender_password']
receiver_email = credentials['receiver_email']

context = ssl.create_default_context()

def send_start_email(train_type, treebank, job_id):
    message = '''\
Hi Dear Furkan,

Mailing from cmpeinspurgpu. Training of type {tt} in treebank {tb} with job ID {job_id} started.

Best,
Furkan
    '''.format(tt=train_type, tb=treebank, job_id=job_id)
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = 'Training started on cmpeinspurgpu!'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    with smtplib.SMTP_SSL('smtp.gmail.com', port, context=context) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

def send_res_email(train_type, treebank, job_id, eval_results):
    ufeats = eval_results['UFeats'].f1
    lemmas = eval_results['Lemmas'].f1
    upos = eval_results['UPOS'].f1
    uas = eval_results['UAS'].f1
    las = eval_results['LAS'].f1
    if train_type in ['feats-only', 'upos_feats']:
        if 'IndFeats' in eval_results.keys():
            ind_feats = eval_results['IndFeats']
            res = f'UFeats: {100*ufeats:.2f}, IndFeats: {ind_feats:.2f}'
        else:
            res = f'UFeats: {100*ufeats:.2f}'
    elif train_type == 'lemma-only':
        res = f'Lemmas: {100*lemmas:.2f}'
    elif train_type == 'pos-only':
        res = f'UPOS: {100*upos:.2f}'
    elif train_type in ['dep-parsing', 'dep-parsing_upos', 'dep-parsing_feats', 'dep-parsing_upos_feats', 'dep-parsing_lemma']:
        res = f'UAS: {100*uas:.2f}, LAS: {100*las:.2f}'

    message = '''\
Hi Dear Furkan,

Mailing from cmpeinspurgpu. Training of type {tt} in treebank {tb} with job ID {job_id} is done.

Results are {res}.

Best,
Furkan
    '''.format(tt=train_type, tb=treebank, job_id=job_id, res=res)
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = 'Training done on cmpeinspurgpu!'
    msg['From'] = sender_email
    msg['To'] = receiver_email

    with smtplib.SMTP_SSL('smtp.gmail.com', port, context=context) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

def send_finish_email():
    message = '''\
Hi Dear Furkan,

Mailing from cmpeinspurgpu. No more jobs to run! 🎉

Best,
Furkan
    '''
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = 'No more jobs to run on cmpeinspurgpu!'
    msg['From'] = sender_email
    msg['To'] = receiver_email

    with smtplib.SMTP_SSL('smtp.gmail.com', port, context=context) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)