// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Cartesian - Euclidean vector space as Lie group

#pragma once

#include "sophus/lie/experimental/lie_group.h"

namespace sophus {

/// Group of 3d Isometries - 3d rotation and translation.
///
/// Alternative names:
///. - 3d Special Euclidean Group, SE(3)
///  - Rigid (body) transformation
///. - 3d pose
///
/// Defining property:
/// A Isometry3 is a distance and handedness preserving transformation. 
///
/// Note: Isometry3 are direct isometries which means that reflections are not 
///       permitted. In more general, all Lie groups in Sophus are 
///.      path-connected. Hence all such transformations preserve handedness, 
///       i.e. the transformation will never map a right-handed reference frame 
///       to a left-handed reference frame.
///
///
/// Number of degrees of freedom: 6
//                                (3 for rotation, 3 for translation)
/// Number of parameters:         7
///                               (4 for quaternion, 3 for translation)
/// Dimension of points:          3
///                               (Group acts on 3d points)
/// Dimension of ambient space:   4
///                               (Isometry3 is a subgroup of the group of 4x4
///                               invertible matrices GL(4).)
///
///
/// Functional description as an action on 3d points:
///
/// 3d rotation followed by a 3d translation.
///
/// Let "point_in_foo" be a point in the foo reference frame. The isometry
/// "bar_from_foo_isometry" maps that point to the reference frame "bar":
///
///.  point_in_foo = bar_from_foo_isometry.rotation() * point_in_foo
///               + bar_from_foo_isometry.translation()
///
///
/// Structural description:
///
/// Represents the orientation and position of a rigid body in 3d.
/// For instance, "world_from_camera_isometry" can be read as "world-anchored
/// camera pose" and describes the position and orientation of the camera in
/// the world reference frame.
///
/// As a Lie group:
///
/// Construction as a subgroup of GL(3): SO(3) x R^3
///.  Semi-direct product of the 3d special orthogonal group (aka 3d rotation 
///.  group) and the abelian group R^3 (aka the 3d vector space).
///
/// ```
///   | R t |
///   | o 1 |
/// ```
///
/// Here, R is an orthogonal 3x3 matrix with determinant of 1, and t is an 
/// element of the 3d Euclidean vector space.
///
/// Lie group properties:
///  - commutative: no
///                 (world_from_robot_isometry * robot_from_camera_isometry
///                  != robot_from_camera_isometry * world_from_robot_isometry)
///  - compact:     no
///                 (Rotations are compact, since we can restrict the rotation
///                 angle to [-pi, pi] without loss of generality. However,
///                 translations are unbounded.)
///
///
template <class Scalar>
using Isometry3 /*aka SE(3) */ = lie::Group<
    lie::SemiDirectProductWithTranslation<Scalar, 3, lie::Rotation3Impl>>;

}  // namespace sophus
