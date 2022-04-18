# Copyright 2022 Petrov, Danil <ddbihbka@gmail.com>. All Rights Reserved.
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''Test suite for brep2graph.configurations'''

import numpy as np

from brep2graph.configurations import simple_edge
from brep2graph.incidence_arrays import IncidenceArrays


def test_simple_edge():
    '''
    CAD model contains 2 faces and edge between them.

    Input:
       F1   |   F2
    Expect:
                  v-----------------------v
          F1 <-> CE1 <-> E1 <-> CE2 <->  F2
          ^-------^--------------^
                  |--------------|
    '''
    graph = simple_edge(face_features=np.zeros((2, 1)),
                        edge_features=np.zeros((1, 1)),
                        coedge_features=np.zeros((2, 1)),
                        incidence_arrays=IncidenceArrays(
                            coedge_to_next=np.zeros((1, ), dtype=np.uint32),
                            coedge_to_mate=np.array([1, 0], dtype=np.uint32),
                            coedge_to_face=np.array([0, 1], dtype=np.uint32),
                            coedge_to_edge=np.array([0, 0], dtype=np.uint32)))
    assert 5 == graph['n_node'][0]
    assert (5, 3) == graph['nodes'].shape

    edges = set(zip(graph['senders'], graph['receivers']))
    assert (0, 3) in edges, 'Face to Coedge'
    assert (0, 4) in edges, 'Mate face to Coedge'
    assert (2, 3) in edges, 'Edge to coedge'
    assert (2, 4) in edges, 'Edge to coedge'
    assert (3, 3) in edges, 'Coedge self-edge'
    assert (3, 4) in edges, 'Coedge to mate coedge'

    assert 16 == graph['n_edge'][0]

