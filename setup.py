import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="psireact",
    version="0.2.1",
    author="Neal Morton",
    author_email="mortonne@gmail.com",
    description="Response time modeling of psychology experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mortonne/psireact",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'seaborn',
        'pymc3',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ]
)
