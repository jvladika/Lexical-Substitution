from contextlib import redirect_stdout
from datetime import datetime

from lexsub_dropout import lexsub_dropout
from load_models import load_data

data = load_data()


'''
Print results either to a file or to standard output.
'''

def print_file():
    now = datetime.now()
    with open('out-' + str(now) + '.txt', 'w') as f:
        with redirect_stdout(f):
            for d in data:
                message, targets = d["message"].lower(), d["keywords"]
                print("\n", message)
                for t in targets:
                    print(t)
                    lexsub_dropout(message, t)

def print_stdout():
    for d in data:
        message, targets = d["message"].lower(), d["keywords"]
        print("\n", message)
        for t in targets:
            print(t)
            lexsub_dropout(message, t)

print_stdout()