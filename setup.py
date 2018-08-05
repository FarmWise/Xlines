from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='xlines',
   version='1.0',
   description='Clustering 2D points in parallel lines',
   license="MIT",
   long_description=long_description,
   author='Sebastien Dubois',
   author_email='sdubois.sebastien@gmail.com',
   url="https://github.com/sds-dubois/Xlines",
   packages=['xlines'],
   install_requires=['numpy', 'sklearn']
)