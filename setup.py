from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('LICENSE', 'r', encoding='utf-8') as f:
    license_text = f.read()

setup(
    name='electrus',
    version='1.1.1',
    author='Pawan kumar',
    author_email='embrakeproject@gmail.com',
    description='Electrus is a lightweight asynchronous & synchronous database module designed for Python.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/embrake/electrus',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords=['database', 'async', 'asynchronous', 'synchronous', 'fast', 'lightweight', 'json'],
    python_requires='>=3.6',
)