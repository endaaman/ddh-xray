import itertools
import invoke
from endaaman.utils import prod_fmt



@invoke.task
def image_all_folds(c):
    cmds = prod_fmt(
        'python classification.py image --model {MODEL} {FEATURES} --fold {FOLD}' \
            ' --name "{{}}_fold{FOLD}" --exp {EXP} --epoch 50',
        {
            'EXP': ['classification_effnet'],
            'MODEL': [f'tf_efficientnet_b{b}' for b in [0]],
            # 'EXP': ['classification_resnet256'],
            # 'MODEL': ['resnet34'],
            'FEATURES': [
                '--lr 0.001 --mode image',
                # '--lr 0.001 --mode integrated',
            ],
            'FOLD': [1, 2, 3, 4, 5, 6],
        }
    )

    # for cmd in cmds:
    #     print(f'RUN: {cmd}')
    #     c.run(cmd)

    # cmds = prod_fmt(
    #     'python classification.py image --model {MODEL} {FEATURES} --fold {FOLD} --name "{{}}_fold{FOLD}" --exp {EXP} -B 16',
    #     {
    #         'EXP': ['classification_resnet_lr'],
    #         'MODEL': ['resnet34'],
    #         'FOLD': [1, 2, 3, 4, 5, 6],
    #         'FEATURES': [
    #             '--lr 0.0001 --mode image',
    #             '--lr 0.0001 --mode additional',
    #             '--lr 0.0001 --mode integrated',
    #         ],
    #     }
    # )

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

@invoke.task
def plot_curves(c):
    for depth in ['b0', 'b4', 'b8']:
        cmds = [
            # metric
            f'python batch.py compare-metric --metric acc --graph box --depth {depth} --noshow',
            f'python batch.py compare-metric --metric acc --graph bar --depth {depth} --noshow',
            f'python batch.py compare-metric --metric f1 --graph box --depth {depth} --noshow',
            f'python batch.py compare-metric --metric f1 --graph bar --depth {depth} --noshow',

            # auc
            f'python batch.py compare-auc --curve roc --graph box --depth {depth} --noshow',
            f'python batch.py compare-auc --curve roc --graph bar --depth {depth} --noshow',
            f'python batch.py compare-auc --curve pr --graph bar --depth {depth} --noshow',
            f'python batch.py compare-auc --curve pr --graph box --depth {depth} --noshow',

            # ROC
            f'python batch.py image-curve-by-folds --curve roc --mode image --depth {depth} --noshow',
            f'python batch.py image-curve-by-folds --curve roc --mode integrated --depth {depth} --noshow',
            f'python batch.py gbm-curve-by-folds --curve roc --depth {depth} --noshow',

            # PR
            f'python batch.py image-curve-by-folds --curve pr --mode image --depth {depth} --noshow',
            f'python batch.py image-curve-by-folds --curve pr --mode integrated --depth {depth} --noshow',
            f'python batch.py gbm-curve-by-folds --curve pr --depth {depth} --noshow',
        ]
        for cmd in cmds:
            print('CMD', cmd)
            c.run(cmd)
