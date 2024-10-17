import ast
import os, sys

x = sys.argv[1]

# assume x is a list of strings, convert to list of strings with ast
x = ast.literal_eval(x)
print(x, type(x))

# command that converts numbers/strings to text strings
txt = lambda x: ('\"' + str(x) + '\"')
print(txt(x), type(txt(x)))
print(str(x))