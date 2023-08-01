import invoke

@invoke.task
def image_all_folds(c):
    for NUM_FEATURES in [0, 8]:
        for FOLD in [1, 2, 3, 4, 5]:
            cmd = f'python classification.py image -F {NUM_FEATURES} --fold {FOLD} --name "{{}}_fold{FOLD}"'
            print(cmd)
            invoke.run(cmd)
