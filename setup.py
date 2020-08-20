from setuptools import setup, find_packages

with open('README.md', encoding='utf8') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='PySwallow',
    version='1.1.0',
    description='A Python Particle Swarm Optimisation Library.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Daniel Kelshaw',
    author_email='daniel.j.kelshaw@gmail.com',
    url='https://github.com/danielkelshaw/PySwallow',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements,
    license='MIT License',
    test_suite='tests'
)
