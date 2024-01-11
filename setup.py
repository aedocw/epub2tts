from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="epub2tts",
    description="Tool to read an epub to audiobook using AI TTS",
    author="Christopher Aedo linkedin.com/in/aedo",
    author_email="doc@aedo.net",
    url="https://github.com/aedocw/epub2tts",
    license="Apache License, Version 2.0",
    version="2.3.9",
    packages=find_packages(),
    install_requires=requirements,
    py_modules=["epub2tts"],
    entry_points={"console_scripts": ["epub2tts = epub2tts:main"]},
)
