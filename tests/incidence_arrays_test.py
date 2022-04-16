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

'''Tests suite for brep2graph.incidence_arrays'''

from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper

from brep2graph.incidence_arrays import build_incidence_arrays


def test_build_incidence_array(body_one_step):
    '''Check build incidence array works without error on the real step file.'''
    body = Compound.load_from_step(body_one_step)
    entity_mapper = EntityMapper(body)

    incidence_arrays = build_incidence_arrays(body, entity_mapper)

    assert incidence_arrays.next.shape != (0, ), 'Expect next not empty'
    msg = 'Expect all incidence arrys have equal shapes.'
    assert (incidence_arrays.next.shape == incidence_arrays.mate.shape and
            incidence_arrays.coedge_to_face.shape == incidence_arrays.coedge_to_edge.shape and
            incidence_arrays.coedge_to_edge.shape == incidence_arrays.next.shape), msg


