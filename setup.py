# Copyright 2021 Petrov, Danil <ddbihbka@gmail.com>. All Rights Reserved.
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

setup(
    name='brep2graph',
    version='0.1.0',
    description='Represent BRep (CAD model structure) as a graph.',
    license='Apache 2.0',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    packages=find_packages(where='src'),
    package_dir={
        'brep2graph': 'src/brep2graph',
    },
    install_requires=[
        'numpy>=1.21',
        'deprecate>=1.0.5',
        'occwl@git+https://github.com/AutodeskAILab/occwl@main',
        'networkx',
    ],
    extras_require={
        'tests': [
            'pytest>=6.2.4',
            'pytest-cov>=2.12',
            'pytest-xdist>=1.34',
            'pytest-regressions',
        ],
    },
    package_data={
        'brep2graph': ['py.typed'],
    },
    include_package_data=True,
    zip_safe=False,
)
