import invoke
from endaaman.utils import prod_fmt

@invoke.task
def image_all_folds(c):
    cmds = prod_fmt(
        'python classification.py image --model {MODEL} -F {NUM_FEATURES} --fold {FOLD} --name "{{}}_fold{FOLD}"',
        {
            'MODEL': [f'tf_efficientnet_b{b}' for b in [0, 4, 8]],
            'NUM_FEATURES': [0, 8],
            'FOLD': [1, 2, 3, 4, 5, 6],
        }
    )

    for cmd in cmds:
        print(f'RUN: {cmd}')
        c.run(cmd)
