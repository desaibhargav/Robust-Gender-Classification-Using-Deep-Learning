from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = ['tensorflow-gpu==1.15.4', 'Keras' ]

setup(name='trainer',
      version='1.0',
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      description='Gender Classification',
      author='Bhargav Desai',
      author_email='desaibhargav98@gmail.com',
      include_package_data=True,
      license='MIT'
)  