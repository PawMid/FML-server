import sys


def progressBar(totalData, dataCount, newline=False):
    filled = (totalData - dataCount) / totalData
    filled = round(filled * 100, 1)
    filNFrac = round(filled)
    bar = ''
    sys.stdout.write('\b' * (len('Progress: |') * 100 + len('| ' + str(filled))))
    bar = bar + '#' * filNFrac + '-' * (100 - filNFrac) + '| ' + str(filled) + '%'
    sys.stdout.write('Progress: |' + ' ' * 100 + '| ' + str(filled) + '\b' * (100 + len('| ' + str(filled))) + bar)
    sys.stdout.flush()
    if filled == 100:
        print(' Done.', end='')
    if newline:
        print()
