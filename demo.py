# -*- coding: UTF-8 -*-
with open('/Users/Peterkwok/Documents/workspace/validation_2016020', 'r')as f:
    result = {}
    for line in f.readlines():
        items = line.split(' ')
        result.setdefault(items[0], []).append(line.strip())
    for k, v in result.items():
        print k
        with open('/Users/Peterkwok/Documents/workspace/validation_02/' + k, 'w')as w:
            w.write('\n'.join(v))
