from setuptools import setup, find_packages
import os

here = os.path.dirname(os.path.abspath(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

with open(os.path.join(here, 'requirements', 'common.txt')) as f:
    required = f.read().splitlines()

setup(
    name='model_fit',

    use_scm_version=True,
    setup_requires=['setuptools_scm'],

    description='Light-weight sugar coating for scipy.optmize.curve_fit',
    long_description=long_description,

    url='https://github.com/nelsond/model_fit',

    author='Nelson Darkwah Oppong',
    author_email='n@darkwahoppong.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='scipy fitting regression optimization',

    packages=find_packages(),
    include_package_data=True,

    install_requires=required
)
