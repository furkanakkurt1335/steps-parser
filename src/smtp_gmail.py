import smtplib, ssl
import json, os

port = 465

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(THIS_DIR, 'gmail-credentials.json'), 'r') as f:
    credentials = json.load(f)
sender_email = credentials['sender_email']
sender_password = credentials['sender_password']
receiver_email = credentials['receiver_email']

def send_email(train_type, treebank, job_id):
    context = ssl.create_default_context()

    message = '''\
Subject: Training done on cmpeinspurgpu!

Hi Dear Furkan,

Mailing from cmpeinspurgpu. Training of type {tt} in treebank {tb} with job ID {job_id} is done.

Best,
Furkan
    '''.format(tt=train_type, tb=treebank, job_id=job_id)

    with smtplib.SMTP_SSL('smtp.gmail.com', port, context=context) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, message)
