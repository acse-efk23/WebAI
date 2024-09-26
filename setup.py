from setuptools import setup, find_packages

setup(
    name='deepdown',
    version='0.1.0',
    author='Ediz Kula',
    author_email='edizferit@gmail.com',
    packages=find_packages(),
    # install_requires=[
    #     'numpy',
    #     'pandas',
    #     'torch',
    # ],
    description='AI model to process subsurface data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/acse-efk23/deep-down',
)