from setuptools import setup, find_packages

setup(
    name='hsc_2024',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchaudio',
        'python-dotenv',
        'soundfile'
    ],
    author='Pascal Makossa',
    author_email='pascal.makossa@tu-dortmund.de',
    python_requires='>=3.9',
)