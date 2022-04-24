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
'''Collection of configuration of different graph respresentaiton of the CAD.

One of the main sources of the configuration is
BRepNet: A topological message passing system for solid models.
https://arxiv.org/pdf/2104.00706.pdf
'''

from typing import Mapping, MutableSequence, Tuple

import numpy as np

from brep2graph.incidence_arrays import IncidenceArrays

Edges = MutableSequence[Tuple[int, int]]


def simple_edge(
    face_features: np.ndarray,
    edge_features: np.ndarray,
    coedge_features: np.ndarray,
    incidence_arrays: IncidenceArrays,
) -> Mapping[str, np.ndarray]:
    '''Simple edge configuration.

     Kernel    |   Faces    |   Edges   |   Coedges
    -------------------------------------------------
    SimpleEdge |   F, MF    |   E       | I, M

    The table shows the description of edges in the graph.
    The graph contains edges between coedges and faces (F)
                                     coedges and mate faces (MF)
                                     coedges and edges (E)
                                     coedges to themself (I)
                                     coedges to mate coedges (M)

    Nodes (vertices) of the graph are entites enumerated
    externally: faces, edges and coedges
    internally: according to index in features table.
    '''
    faces_num = face_features.shape[0]
    edges_num = edge_features.shape[0]
    coedges_num = coedge_features.shape[0]

    face_to_node = np.arange(faces_num)
    edge_to_node = np.arange(edges_num) + faces_num
    coedge_to_node = np.arange(coedges_num) + faces_num + edges_num

    edges: Edges = []
    # Faces
    _f(incidence_arrays.coedge_to_face, coedge_to_node, face_to_node, edges)
    _mf(incidence_arrays.coedge_to_mate, incidence_arrays.coedge_to_face,
        coedge_to_node, face_to_node, edges)
    # Edges
    _e(incidence_arrays.coedge_to_edge, coedge_to_node, edge_to_node, edges)
    # Coedges
    _i(np.arange(coedges_num), coedge_to_node, edges)
    _m(incidence_arrays.coedge_to_mate, coedge_to_node, edges)

    return _create_graph(face_features, edge_features, coedge_features, edges)


def _create_graph(face_features: np.ndarray, edge_features: np.ndarray,
                  coedge_features: np.ndarray,
                  edges: Edges) -> Mapping[str, np.ndarray]:
    '''Helper method to create the graph of given nodes (as features) and edges.'''
    senders = []
    receivers = []
    for (f, t) in edges:
        senders.append(f)
        receivers.append(t)

    assert len(senders) == len(receivers)

    return {
        'face_features': face_features,
        'edge_features': edge_features,
        'coedge_features': coedge_features,
        'senders': np.array(senders),
        'receivers': np.array(receivers),
    }


def _f(
    coedge_to_face: np.ndarray,
    coedge_to_node: np.ndarray,
    face_to_node: np.ndarray,
    edges: Edges,
):
    '''Returns edges from coedges to faces.
    Mutates given `edges` sequence.
    '''
    for coedge_ix, face_ix in enumerate(coedge_to_face):
        coedge = coedge_to_node[coedge_ix]
        face = face_to_node[face_ix]

        _add_undirect_edge(coedge, face, edges)


def _mf(coedge_to_mate: np.ndarray, coedge_to_face: np.ndarray,
        coedge_to_node: np.ndarray, face_to_node: np.ndarray, edges: Edges):
    '''Returns edges from coedges to mate faces.'''
    for coedge_ix, _ in enumerate(coedge_to_face):
        coedge = coedge_to_node[coedge_ix]
        mate_face = face_to_node[coedge_to_face[coedge_to_mate[coedge_ix]]]

        _add_undirect_edge(coedge, mate_face, edges)


def _e(coedge_to_edge: np.ndarray, coedge_to_node: np.ndarray,
       edge_to_node: np.ndarray, edges: Edges):
    '''Returns edges from coedges to edges.'''
    for coedge_ix, edge_ix in enumerate(coedge_to_edge):
        coedge = coedge_to_node[coedge_ix]
        edge = edge_to_node[edge_ix]

        _add_undirect_edge(coedge, edge, edges)


def _i(coedges_to_coedges: np.ndarray, coedge_to_node: np.ndarray,
       edges: Edges):
    '''Returns edges from coedges to coedges (self-edges)'''
    for coedge_ix in coedges_to_coedges:
        coedge = coedge_to_node[coedge_ix]

        # Here we have self edge
        edges.append((coedge, coedge))


def _m(coedge_to_mate: np.ndarray, coedge_to_node: np.ndarray, edges: Edges):
    '''Returns edges from coedges to mate coedges.'''
    for coedge_ix, mate_coedge_ix in enumerate(coedge_to_mate):
        coedge = coedge_to_node[coedge_ix]
        mate_coedge = coedge_to_node[mate_coedge_ix]

        # Here because we are going to iterate over all coedges
        # no reason to add undirect edge (or it will double edges)
        edges.append((coedge, mate_coedge))


def _add_undirect_edge(from_, to_, edges):
    edges.append((from_, to_))
    edges.append((to_, from_))
