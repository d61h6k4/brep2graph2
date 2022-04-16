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
'''Tests suite for brep2graph.features'''

import numpy as np

from occwl.compound import Compound

from brep2graph.features import features_from_body, edge_features_from_body, face_features_from_body


def test_features_from_body(ndarrays_regression, body_one_step):
    '''Check build incidence array works without error on the real step file.'''
    body = Compound.load_from_step(body_one_step)

    features = features_from_body(body)

    ndarrays_regression.check({'features': features})


def test_face_features_from_body(body_one_step):
    '''Validate the properties of the face features.'''

    def _check(cond, msg):
        assert np.all(cond), f'{msg}, but {features[np.where(~cond)]}'

    body = Compound.load_from_step(body_one_step)

    features = face_features_from_body(body)

    _check((features[:, 0] > 0), 'Area feature should be no negative')
    _check((features[:, 1] >= 0) & (features[:, 1] <= 1),
           'Plane is True/False feature')
    _check((features[:, 2] >= 0) & (features[:, 2] <= 1),
           'Cylinder is True/False feature')
    _check((features[:, 3] >= 0.) & (features[:, 3] <= 1),
           'Cone is True/False feature')
    _check((features[:, 4] >= 0) & (features[:, 4] <= 1),
           'Sphere is True/False feature')
    _check((features[:, 5] >= 0) & (features[:, 5] <= 1),
           'Torus is True/False feature')
    _check((features[:, 6] >= 0) & (features[:, 6] <= 1),
           'Bezier is True/False feature')
    _check((features[:, 7] >= 0) & (features[:, 7] <= 1),
           'BSpline is True/False feature')
    _check((features[:, 8] >= 0) & (features[:, 8] <= 1),
           'Revolution is True/False feature')
    _check((features[:, 9] >= 0) & (features[:, 9] <= 1),
           'Extrusion is True/False feature')
    _check((features[:, 10] >= 0) & (features[:, 10] <= 1),
           'Offset is True/False feature')
    _check((features[:, 11] >= 0) & (features[:, 11] <= 1),
           'Other is True/False feature')
    _check((features[:, 12] >= 0) & (features[:, 12] <= 1),
           'Unknown is True/False feature')


def test_edge_features_from_body(body_one_step):
    '''Validate the properties of the edge features.'''

    def _check(cond, msg):
        assert np.all(cond), f'{msg}, but {features[np.where(~cond)]}'

    body = Compound.load_from_step(body_one_step)

    features = edge_features_from_body(body)

    _check((features[:, 0] >= 0) & (features[:, 0] <= 1),
           'Concavity is True/False feature')
    _check((features[:, 1] >= 0) & (features[:, 1] <= 1),
           'Convexity is True/False feature')
    _check((features[:, 2] >= 0) & (features[:, 2] <= 1),
           'Smoothity is True/False feature')
    _check(features[:, 3] >= 0., 'Length should be positive number')
    _check((features[:, 4] >= 0) & (features[:, 4] <= 1),
           'Closed curve is True/False feature')
    _check((features[:, 5] >= 0) & (features[:, 5] <= 1),
           'Periodic edge is True/False feature')
    _check((features[:, 6] >= 0) & (features[:, 6] <= 1),
           'Rational edge is True/False feature')
    _check((features[:, 7] >= 0) & (features[:, 7] <= 1),
           'Linear edge is True/False feature')
    _check((features[:, 8] >= 0) & (features[:, 8] <= 1),
           'Circular is True/False feature')
    _check((features[:, 9] >= 0) & (features[:, 9] <= 1),
           'Ellipse is True/False feature')
    _check((features[:, 10] >= 0) & (features[:, 10] <= 1),
           'Hyperbola is True/False feature')
    _check((features[:, 11] >= 0) & (features[:, 11] <= 1),
           'Parabola is True/False feature')
    _check((features[:, 12] >= 0) & (features[:, 12] <= 1),
           'Bezier is True/False feature')
    _check((features[:, 13] >= 0) & (features[:, 13] <= 1),
           'BSpline is True/False feature')
    _check((features[:, 14] >= 0) & (features[:, 14] <= 1),
           'Offset is True/False feature')
    _check((features[:, 15] >= 0) & (features[:, 15] <= 1),
           'Other is True/False feature')
    _check((features[:, 16] >= 0) & (features[:, 16] <= 1),
           'Unknown is True/False feature')
