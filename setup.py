import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graddy",                     # This is the name of the package
    version="0.0.1",                        # The initial release version
    author="Quadeer Shaikh | Yash Shah | Ishita Gupta",         # Full name of the author
    author_email='yash161101@gmail.com', # Author Email ID
    description="A tool for generating class activation maps in deep learning models.", # Short Description of the module
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    # py_modules=["pdistributions"],             # Name of the python package
    package_dir={'':'src'},     # Directory of the source code of the package
    install_requires=['tensorflow','numpy','matplotlib', 'keras']                     # Install other dependencies if any
)
