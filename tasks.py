import itertools
import invoke
from endaaman.utils import prod_fmt



@invoke.task
def image_all_folds(c):
    # cmds = prod_fmt(
    #     'python classification.py image --model {MODEL} {FEATURES} --fold {FOLD} --name "{{}}_fold{FOLD}" --exp {EXP}',
    #     {
    #         'EXP': ['classification_effnet'],
    #         'MODEL': [f'tf_efficientnet_b{b}' for b in [0, 4]],
    #         # 'EXP': ['classification_resnet256'],
    #         # 'MODEL': ['resnet34'],
    #         'FEATURES': ['', '-F'],
    #         'FOLD': [1, 2, 3, 4, 5, 6],
    #     }
    # )

    # for cmd in cmds:
    #     print(f'RUN: {cmd}')
    #     c.run(cmd)

    cmds = prod_fmt(
        'python classification.py image --model {MODEL} {FEATURES} --fold {FOLD} --name "{{}}_fold{FOLD}" --exp {EXP} -B 16',
        {
            'EXP': ['classification_resnet_freeze'],
            'MODEL': ['resnet34', 'resnet50'],
            'FOLD': [1, 2, 3, 4, 5, 6],
            'FEATURES': [
                '--lr 0.001 --mode image',
                '--lr 0.0001 --mode additional',
                '--lr 0.0001 --mode integrated',
            ],
        }
    )

    for cmd in cmds:
        print(f'RUN: {cmd}')
        c.run(cmd)


@invoke.task
def fold1(c):
    cmds = prod_fmt(
        'python classification.py image --model {MODEL} {FEATURES} --fold {FOLD} --name "{{}}_fold{FOLD}" --exp {EXP} -B 16',
        {
            'EXP': ['classification_resnet_freeze'],
            'MODEL': ['resnet50'],
            'FOLD': [1],
            'FEATURES': [
                '--lr 0.001 --mode image',
                '--lr 0.0001 --mode additional',
                '--lr 0.0001 --mode integrated',
            ],
        }
    )

    for cmd in cmds:
        print(f'RUN: {cmd}')
        c.run(cmd)

@invoke.task
def p(c):
    cmds = prod_fmt('fold:{fold} feature:{feature}', {'fold': [0, 1], 'feature': [0, 8]})
    print(cmds)
