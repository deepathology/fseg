import setuptools

with open('README.md', mode='r', encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    name='fseg',
    version='1.0.0',
    author='Ofir Hadar, Jacob Gildenblat',
    author_email='jacob.gildenblat@gmail.com',
    description='Unsupervised segmentation with DFF',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/deepathology/dff_seg',
    project_urls={
        'Bug Tracker': 'https://github.com/deepathology/dff_seg/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    packages=setuptools.find_packages(
        exclude=["*notebooks*"]),
    python_requires='>=3.9',
    install_requires=requirements)
