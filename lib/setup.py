from distutils.core import setup
from distutils.util import convert_path
from fnmatch import fnmatchcase
import os

__author__ = 'avive'

NAME = 'traxDL'
MAINTAINER = "Traxretail"
MAINTAINER_EMAIL = "aviv@traxretail.com"
DESCRIPTION = ('Deep learning library for trax applications.')

AUTHOR = "Traxretail"
AUTHOR_EMAIL = "aviv@traxretail.com"
PLATFORMS = ["Windows", "Linux"]
MAJOR = 0
MINOR = 1
MICRO = 0
SUFFIX = ""  # Should be blank except for rc's, betas, etc.
ISRELEASED = False

VERSION = '%d.%d.%d%s' % (MAJOR, MINOR, MICRO, SUFFIX)


def find_packages(where='.', exclude=()):
    out = []
    stack = [(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where, name)
            if ('.' not in name and os.path.isdir(fn) and
                    os.path.isfile(os.path.join(fn, '__init__.py'))
                ):
                out.append(prefix + name)
                stack.append((fn, prefix + name + '.'))
    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out


def do_setup():
    setup(name=NAME,
          version=VERSION,
          description=DESCRIPTION,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          platforms=PLATFORMS,
          packages=find_packages(where=os.path.dirname(os.path.abspath(__file__))),
          install_requires=['theano>=0.7.0'])

if __name__ == "__main__":
    do_setup()
