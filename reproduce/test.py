import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

@hidden
def print_something():
    print('abcxyz')

# with HiddenPrints():
#     with open('/home/giang/Desktop/temp.txt', 'w') as f:
#         print('abc', file = f)

# print('xyz')

print_something()
print('mlem')