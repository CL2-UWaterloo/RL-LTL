import numpy as np
from LTL import *

x = [False, True, False, True, False, True]
y = [True, True, True, False, False, True]

predicates={'a':[12], 'b':[2], 'c':[17], 'd':[3]}

trajectory = [0,0,0,1]
trajectory += trajectory
trajectory += trajectory
trajectory += trajectory

f1 = ('~', (None, 'd'))
f2 = ('->', ('/\\', (None, 'b'), ('~', ('>', (None, 'b')))), ('>', ('%', ('~', (None, 'b')), ('\\/', (None, 'a'), (None, 'c')))))
f3 = ('->', (None, 'a'),
     ('>', ('%', ('~', (None, 'a')), (None, 'b'))))
f4 = ('->',
    ('/\\', ('~', (None, 'b')), ('/\\', ('>', (None, 'b')), ('~', ('>', ('>', (None, 'b')))))),
    ('%', ('~', (None, 'a')), (None, 'c')))
f5 = ('->', (None, 'c'), ('%', ('~', (None, 'a')), (None, 'b')))
f6 = ('->', ('/\\', (None, 'b'), ('>', (None, 'b'))), ('<>', (None, 'a')))

f5_6 = ('/\\', f5, f6)
f4_5_6 = ('/\\', f4, f5_6)
f3_4_5_6 = ('/\\', f3, f4_5_6)
f2_3_4_5_6 = ('/\\', f2, f3_4_5_6)
f1_2_3_4_5_6 = ('/\\', f1, f2_3_4_5_6)
formula = ("[]", f1_2_3_4_5_6)

x = check_LTL(formula, trajectory, predicates)

print(len(str(formula)))


print(len(str(ast)))

for i in range(len(str(ast))):
    if str(ast)[i]!=str(formula)[i]:
        print(str(ast)[i],str(formula)[i])