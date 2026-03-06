import os
import traceback

import main_part1
#import main_part2
#import main_part3


def _safe_run(name, fn):
    print(f'\n===== Running {name} =====')
    try:
        fn()
    except Exception as exc:
        print(f'{name} failed: {exc}')
        traceback.print_exc()


if __name__ == '__main__':
    _safe_run('Partie I', main_part1.run)
    #_safe_run('Partie III', main_part3.run)
    #_safe_run('Partie II', main_part2.run)
