import email

from email import policy
from itertools import groupby


def parse_mail_content(content: str) -> dict:
    target = {}
    msg = email.message_from_string(content, policy=policy.SMTPUTF8)

    target['headers'] = dict([(key.lower(), list(v[1] for v in group)) for (key,group) in groupby(msg.items(), lambda x: x[0])])
    target['text'] = msg.get_content().strip()
    target['type'] = msg.get_content_type()

    return(target)
