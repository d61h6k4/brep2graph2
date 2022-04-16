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
'''Build incidence arrays.

Here is the diagram that describes the relationships:
https://github.com/AutodeskAILab/BRepNet/blob/master/docs/img/BRepTopologicalWalkv02.png
'''


from typing import NamedTuple

import numpy as np

from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper


class IncidenceArrays(NamedTuple):
    '''Incidence arrays encode relationship between BRep entities (face, edge and coedge).
    '''
    next: np.ndarray
    mate: np.ndarray
    coedge_to_face: np.ndarray
    coedge_to_edge: np.ndarray


def build_incidence_arrays(body: Compound,
                           entity_mapper: EntityMapper) -> IncidenceArrays:
    '''Build incidence arrays for the given body.

    To use/store index (number) as an identificator of an entity we use global registry (entity_mapper).
    All relationship described within coedges.
    '''
    coedges_num = len(entity_mapper.oriented_edge_map)

    # Initializer used to initialize an incidence arrays
    # we initialize with impossible value to use this knowledge
    # for validation.
    initializer = lambda: np.zeros(
        (coedges_num, ), dtype=np.uint32) + coedges_num

    next_ = initializer()
    mate = initializer()
    coedge_to_edge = initializer()
    coedge_to_face = initializer()

    # Create the next, previous and mate permutations.
    for wire in body.wires():

        first_coedge_index = None
        previous_coedge_index = None
        for coedge in wire.ordered_edges():
            coedge_index = entity_mapper.oriented_edge_index(coedge)
            # coedge is an edge with additional data (orientation)
            edge_index = entity_mapper.edge_index(coedge)

            # Set up the mating coedge
            mating_coedge = coedge.reversed_edge()
            if entity_mapper.oriented_edge_exists(mating_coedge):
                mating_coedge_index = entity_mapper.oriented_edge_index(
                    mating_coedge)
            else:
                # If a coedge has no mate then we mate it to itself.
                # This typically happens at the poles of sphere.
                mating_coedge_index = coedge_index
            mate[coedge_index] = mating_coedge_index

            msg = f'Coedge and it\'s mate should share the edge. {coedge_index=} {mating_coedge_index=} {edge_index=}'
            assert edge_index == entity_mapper.edge_index(mating_coedge), msg
            coedge_to_edge[coedge_index] = edge_index
            coedge_to_edge[mating_coedge_index] = edge_index

            if first_coedge_index is None:
                first_coedge_index = coedge_index
            else:
                next_[previous_coedge_index] = coedge_index
            previous_coedge_index = coedge_index

        # Close the loop (wire)
        next_[previous_coedge_index] = first_coedge_index

    for face in body.faces():
        face_index = entity_mapper.face_index(face)
        for wire in body.wires_from_face(face):
            for coedge in wire.ordered_edges():
                coedge_index = entity_mapper.oriented_edge_index(coedge)
                coedge_to_face[coedge_index] = face_index

    assert np.all(next_ < coedges_num), 'next contains a coedge without next'
    assert np.all(mate < coedges_num), 'mate contains a coedge without mate'
    assert np.all(coedge_to_face < coedges_num), 'coedge_to_face contains coedge without relation to face'
    assert np.all(coedge_to_edge < coedges_num), 'next contains coedge without relation to edge'

    return IncidenceArrays(next=next_,
                           mate=mate,
                           coedge_to_face=coedge_to_face,
                           coedge_to_edge=coedge_to_edge)
