from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="hindi_english_translator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Translator Team",
    author_email="nmaurya.engineer@gmail.com",
    description="A context-aware Hindi-English translator",
    keywords="translation, nlp, hindi, english",
    python_requires=">=3.7",
) 