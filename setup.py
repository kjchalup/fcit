from distutils.core import setup

def read(fnam):
    """ Utility function for README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'dtit',
  packages = ['.'], # this must be the same as the name above
  version = '1.0',
  description = 'A decision-tree based conditional independence test',
  author = 'Krzysztof Chalupka',
  author_email = 'kjchalup@caltech.edu',
  license='MIT',
  install_requires=[
      'numpy>=1.12.1',
      'sklearn>=0.0',
      'scipy>=0.19.0'],
  url = 'https://github.com/kjchalup/dtit',
  download_url = 'https://github.com/kjchalup/dtit/archive/1.0tar.gz',
  keywords = ['graphical models', 'statistics',
      'machine learning', 'decision tree']
)

