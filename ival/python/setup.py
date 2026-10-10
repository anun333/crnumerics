"""Only for a wheel that carries libival.so.0 (build-wheel.sh): it is for one platform, and the package is Python
and ctypes, so any Python 3 can use it. The distribution reports binaries, which files the package as platlib,
and the tag says py3-none-<platform>."""
import os

from setuptools import setup
from setuptools.dist import Distribution

bundled = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "ival", "libival.so.0"))
kw = {}
if bundled:
    try:
        from setuptools.command.bdist_wheel import bdist_wheel
    except ImportError:   # setuptools before 70.1
        from wheel.bdist_wheel import bdist_wheel

    class BinaryDistribution(Distribution):
        def has_ext_modules(self):
            return True

    class PlatformWheel(bdist_wheel):
        def get_tag(self):
            return "py3", "none", super().get_tag()[2]

    kw = {"distclass": BinaryDistribution, "cmdclass": {"bdist_wheel": PlatformWheel}}

setup(**kw)
