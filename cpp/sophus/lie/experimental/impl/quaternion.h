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
#include "sophus/lie/experimental/lie_group_concept.h"

// Include only the selective set of Eigen headers that we need.
// This helps when using Sophus with unusual compilers, like nvcc.
#include <Eigen/src/Geometry/OrthoMethods.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/src/Geometry/RotationBase.h>

namespace sophus {
namespace lie {

template <class TScalar>
class QuaternionNumberImpl {
 public:
  using Scalar = TScalar;

  static auto multiplication(
      Eigen::Vector<Scalar, 4> const& a, Eigen::Vector<Scalar, 4> const& b)
      -> Eigen::Vector<Scalar, 4> {
    return Eigen::Vector<Scalar, 4>(
        a.w() * b.w() - a.x() * b.x() - a.y() * b.y() - a.z() * b.z(),
        a.w() * b.x() + a.x() * b.w() + a.y() * b.z() - a.z() * b.y(),
        a.w() * b.y() + a.y() * b.w() + a.z() * b.x() - a.x() * b.z(),
        a.w() * b.z() + a.z() * b.w() + a.x() * b.y() - a.y() * b.x());
  }

  static auto conjugate(Eigen::Vector<Scalar, 4> const& a)
      -> Eigen::Vector<Scalar, 4> {
    return Eigen::Vector<Scalar, 4>(-a.x(), -a.y(), -a.z(), a.w());
  }
};

}  // namespace lie
}  // namespace sophus
