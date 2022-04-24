# Copyright 2022 Petrov, Danil <ddbihbka@gmail.com>. All Rights Reserved.
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
'''Extract features from the BRep shape.

Main source is BRepNet paper:
@inproceedings{lambourne2021brepnet,
    author    = {Lambourne, Joseph G. and Willis, Karl D.D. and Jayaraman,
                 Pradeep Kumar and Sanghi, Aditya and Meltzer, Peter and Shayani, Hooman},
    title     = {BRepNet: A Topological Message Passing System for Solid Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {12773-12782}
}
'''

from typing import Iterable, Optional

import numpy as np

from occwl.entity_mapper import EntityMapper
from occwl.edge import Edge
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
from occwl.compound import Compound
from occwl.face import Face


def features_from_body(body: Compound, entity_mapper: EntityMapper) -> np.ndarray:
    '''Returns all features of all type of entities of the body.'''
    face_features = face_features_from_body(body, entity_mapper)
    edge_features = edge_features_from_body(body, entity_mapper)
    coedge_features = coedge_features_from_body(body, entity_mapper)

    features = np.block([
        [
            face_features,
            np.zeros((face_features.shape[0],
                      edge_features.shape[1] + coedge_features.shape[1]))
        ],
        [
            np.zeros((edge_features.shape[0], face_features.shape[1])),
            edge_features,
            np.zeros((edge_features.shape[0], coedge_features.shape[1])),
        ],
        [
            np.zeros((coedge_features.shape[0],
                      face_features.shape[1] + edge_features.shape[1])),
            coedge_features,
        ],
    ])

    return features


def face_features_from_body(body: Compound, entity_mapper: EntityMapper) -> np.ndarray:
    '''Returns face features of the body'''
    face_features = []
    for ix, face in enumerate(body.faces()):
        assert ix == entity_mapper.face_index(face)
        face_features.append(features_from_face(face))
    return np.stack(face_features)


def edge_features_from_body(body: Compound, entity_mapper: EntityMapper) -> np.ndarray:
    '''Returns edge features of the body'''

    edge_features = []
    for ix, edge in enumerate(body.edges()):
        assert ix == entity_mapper.edge_index(edge)
        edge_features.append(
            features_from_edge(edge, body.faces_from_edge(edge)))
    return np.stack(edge_features)


def coedge_features_from_body(body: Compound, entity_mapper: EntityMapper) -> np.ndarray:
    '''Returns coedge features of the body.'''
    coedge_features = []

    ix = 0
    for wire in body.wires():
        for coedge in wire.ordered_edges():
            assert ix == entity_mapper.oriented_edge_index(coedge)
            coedge_features.append(features_from_coedge(coedge))
            ix += 1

    return np.stack(coedge_features)


def features_from_face(face: Face) -> np.ndarray:
    '''Returns features of the face.'''
    return np.array([
        face_area_feature(face),
        plane_face_feature(face),
        cylinder_face_feature(face),
        cone_face_feature(face),
        sphere_face_feature(face),
        torus_face_feature(face),
        bezier_face_feature(face),
        bspline_face_feature(face),
        revolution_face_feature(face),
        extrusion_face_feature(face),
        offset_face_feature(face),
        other_face_feature(face),
        unknown_face_feature(face),
    ],
                    dtype=np.float32)


def features_from_edge(edge: Edge, faces: Iterable[Face]) -> np.ndarray:
    '''Returns features of the edge.'''
    convexity = find_edge_convexivity(edge, faces)
    return np.array([
        concavity_feature(convexity),
        convexity_feature(convexity),
        smoothity_feature(convexity),
        edge_length_feature(edge),
        closed_curve_features(edge),
        periodic_edge_feature(edge),
        rational_edge_feature(edge),
        linear_edge_feature(edge),
        circular_edge_feature(edge),
        ellipse_edge_feature(edge),
        hyperbola_edge_feature(edge),
        parabola_edge_feature(edge),
        bezier_edge_feature(edge),
        bspline_edge_feature(edge),
        offset_edge_feature(edge),
        other_edge_feature(edge),
        unknown_edge_feature(edge)
    ],
                    dtype=np.float32)


def features_from_coedge(coedge: Edge) -> np.ndarray:
    '''Returns features of the coedge.'''
    return np.array([
        reversed_edge_feature(coedge),
    ], dtype=np.float32)


def face_area_feature(face: Face) -> np.float32:
    '''Area of the face.'''
    return np.float32(face.area())


def plane_face_feature(face: Face) -> np.float32:
    '''Is face surface type plane?'''
    return np.float32(face.surface_type() == 'plane')


def cylinder_face_feature(face: Face) -> np.float32:
    '''Is face surface type cylinder?'''
    return np.float32(face.surface_type() == 'cylinder')


def cone_face_feature(face: Face) -> np.float32:
    '''Is face surface type cone?'''
    return np.float32(face.surface_type() == 'cone')


def sphere_face_feature(face: Face) -> np.float32:
    '''Is face surface type sphere?'''
    return np.float32(face.surface_type() == 'sphere')


def torus_face_feature(face: Face) -> np.float32:
    '''Is face surface type torus?'''
    return np.float32(face.surface_type() == 'torus')


def bezier_face_feature(face: Face) -> np.float32:
    '''Is face surface type bezier?'''
    return np.float32(face.surface_type() == 'bezier')


def bspline_face_feature(face: Face) -> np.float32:
    '''Is face surface type bspline?'''
    return np.float32(face.surface_type() == 'bspline')


def revolution_face_feature(face: Face) -> np.float32:
    '''Is face surface type revolution?'''
    return np.float32(face.surface_type() == 'revolution')


def extrusion_face_feature(face: Face) -> np.float32:
    '''Is face surface type extrusion?'''
    return np.float32(face.surface_type() == 'extrusion')


def offset_face_feature(face: Face) -> np.float32:
    '''Is face surface type offset?'''
    return np.float32(face.surface_type() == 'offset')


def other_face_feature(face: Face) -> np.float32:
    '''Is face surface type other?'''
    return np.float32(face.surface_type() == 'other')


def unknown_face_feature(face: Face) -> np.float32:
    '''Is face surface type unknown?'''
    return np.float32(face.surface_type() == 'unknown')


def find_edge_convexivity(edge: Edge,
                          faces: Iterable[Face]) -> Optional[EdgeConvexity]:
    '''Edge convexivity.'''
    edge_data = EdgeDataExtractor(edge,
                                  list(faces),
                                  use_arclength_params=False)
    if not edge_data.good:
        # This is the case where the edge is a pole of a sphere
        return None
    # defines the smoothnes relative to this angle
    angle_tol_rads = 0.0872664626  # 5 degrees
    convexity = edge_data.edge_convexity(angle_tol_rads)
    return convexity


def concavity_feature(convexity: Optional[EdgeConvexity]) -> np.float32:
    '''Is edge concave?'''
    return np.float32(convexity is not None
                      and convexity == EdgeConvexity.CONCAVE)


def convexity_feature(convexity: Optional[EdgeConvexity]) -> np.float32:
    '''Is edge concave?'''
    return np.float32(convexity is not None
                      and convexity == EdgeConvexity.CONVEX)


def smoothity_feature(convexity: Optional[EdgeConvexity]) -> np.float32:
    '''Is edge concave?'''
    return np.float32(convexity is not None
                      and convexity == EdgeConvexity.SMOOTH)


def edge_length_feature(edge: Edge) -> np.float32:
    '''The lenght of the edge.'''
    return np.float32(edge.length())


def closed_curve_features(edge: Edge) -> np.float32:
    '''Is curve of the edge closed?'''
    return np.float32(edge.has_curve() and edge.closed_curve())


def periodic_edge_feature(edge: Edge) -> np.float32:
    '''Is the edge periodic?'''
    return np.float32(edge.periodic())


def rational_edge_feature(edge: Edge) -> np.float32:
    '''Is the edge rational?'''
    return np.float32(edge.rational())


def linear_edge_feature(edge: Edge) -> np.float32:
    '''Is edge linear?'''
    return np.float32(edge.curve_type() == 'line')


def circular_edge_feature(edge: Edge) -> np.float32:
    '''Is edge circular?'''
    return np.float32(edge.curve_type() == 'circular')


def ellipse_edge_feature(edge: Edge) -> np.float32:
    '''Is edge ellipse?'''
    return np.float32(edge.curve_type() == 'ellipse')


def hyperbola_edge_feature(edge: Edge) -> np.float32:
    '''Is edge hyperbola?'''
    return np.float32(edge.curve_type() == 'hyperbola')


def parabola_edge_feature(edge: Edge) -> np.float32:
    '''Is edge parabola?'''
    return np.float32(edge.curve_type() == 'parabola')


def bezier_edge_feature(edge: Edge) -> np.float32:
    '''Is edge bezier?'''
    return np.float32(edge.curve_type() == 'bezier')


def bspline_edge_feature(edge: Edge) -> np.float32:
    '''Is edge bspline?'''
    return np.float32(edge.curve_type() == 'bspline')


def offset_edge_feature(edge: Edge) -> np.float32:
    '''Is edge offset?'''
    return np.float32(edge.curve_type() == 'offset')


def other_edge_feature(edge: Edge) -> np.float32:
    '''Is edge other?'''
    return np.float32(edge.curve_type() == 'other')


def unknown_edge_feature(edge: Edge) -> np.float32:
    '''Is edge unknown?'''
    return np.float32(edge.curve_type() == 'unknown')


def reversed_edge_feature(edge: Edge) -> np.float32:
    '''Is the edge reversed?'''
    return np.float32(edge.reversed())
