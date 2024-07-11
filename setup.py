from setuptools import setup, find_packages

setup(
    name='ref_l4',
    version='1.0',
    packages=find_packages(),
    py_modules=['dataloader', 'evaluation', 'overlaps'],
    install_requires=[
        # Add any dependencies your package needs here
        # e.g., 'numpy', 'pandas', 'scikit-learn'
        'torch', 'datasets', 'pillow', 'statistics', 'pandas', 'datasets'
    ],
    entry_points={
        'console_scripts': [
            # Add any command line scripts here
            # e.g., 'mycommand = mypackage.module:function'
        ],
    },
    author='Jierun Chen',
    description='Evaluation code for Ref-L4, a new REC benchmark in the LMM era.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/JierunChen/Ref-L4',  # Replace with your package's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT',
    python_requires='>=3.6',
)