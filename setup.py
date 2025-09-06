from setuptools import setup

setup(
    name='cv3',
    version='1.3.1',
    packages=['cv3'],
    url='https://github.com/gorodion/cv3',
    license='Apache 2.0',
    author='gorodion',
    author_email='domonion@list.ru',
    description='Pythonic cv2',
    install_requires=[
        'numpy>=1.19.5',
        'opencv-python>=4.2.0.34'
    ],
    project_urls={
        "Documentation": "https://cv3.readthedocs.io/en/latest/",
        "Source": "https://github.com/gorodion/cv3",
    },
)
