from setuptools import setup, find_packages

setup(
    name='sffm',
    version='0.1.0',
    description='Sequential Financial Foundation Model with Gaussian Process Uncertainty & Fairness',
    author='Farah Qoonita Syuhaila',
    author_email='farahqoonita2@gmail.com',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'gpytorch>=1.11',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
    ],
)
