from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='epub-to-TTS-app',
    version='1.0.3',
    packages=find_packages(),
    install_requires=requirements,
    py_modules=['epub2tts'],
    entry_points={
        'console_scripts': [
            'epub2tts = epub2tts:main'
        ]
    }
)
