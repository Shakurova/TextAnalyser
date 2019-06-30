import os
import setuptools

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def get_readme():
    with open("README.md", "r") as fh:
        long_description = fh.read()
        return long_description

def get_requirements():
    """ Parses requirements from requirements.txt """
    reqs_path = os.path.join(__location__, 'requirements.txt')
    with open(reqs_path, encoding='utf8') as f:
        reqs = [line.strip() for line in f if not line.strip().startswith('#')]

    names = []
    links = []
    for req in reqs:
        if '://' in req:
            links.append(req)
        else:
            names.append(req)

    return {'install_requires': names, 'dependency_links': links}

setuptools.setup(
    name="text-analyser",
    version="0.0.1",
    author="Lena Shakurova",
    author_email="lenashakurova.work@gmail.com",
    description="NLP components for extracting information from text.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Shakurova/TextAnalyser/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    **get_requirements()
)
