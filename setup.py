from setuptools import setup, find_packages

setup(
    name='inclinet',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.4',
    ],
    author='Adarsh Dubey',
    author_email='dubeyadarshmain@gmail.com',
    description='A simple neural network implementation using Python and Numpy',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/inclinedadarsh/inclinet',  # Update with your project's URL if available
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
