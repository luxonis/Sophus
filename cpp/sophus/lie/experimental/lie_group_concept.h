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
#include "sophus/common/concept_utils.h"
#include "sophus/common/types.h"

namespace sophus {
namespace lie {

template <class T>
concept LieGroupImplConcept =
    (T::kPointDim == 2 || T::kPointDim == 3) &&  // 2d or 3d points
    (T::kPointDim == T::kAmbientDim  // inhomogeneous point representation
     ||
     T::kPointDim + 1 == T::kAmbientDim)  // or homogeneous point representation
    && requires(
           T g,
           Eigen::Vector<typename T::Scalar, T::kDof> tangent,
           Eigen::Vector<typename T::Scalar, T::kPointDim> point,
           Eigen::Vector<typename T::Scalar, T::kNumParams> params,
           Eigen::Matrix<typename T::Scalar, T::kAmbientDim, T::kAmbientDim>
               matrix,
           Eigen::Matrix<typename T::Scalar, T::kDof, T::kDof> adjoint) {
  // constructors and factories
  {
    T::identityParams()
    } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kNumParams>>;

  { T::areParamsValid(params) } -> ConvertibleTo<sophus::Expected<Success>>;

  // Manifold / Lie Group concepts

  {
    T::exp(tangent)
    } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kNumParams>>;

  {
    T::log(params)
    } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kDof>>;

  {
    T::hat(tangent)
    } -> ConvertibleTo<
        Eigen::Matrix<typename T::Scalar, T::kAmbientDim, T::kAmbientDim>>;

  {
    T::vee(matrix)
    } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kDof>>;

  {
    T::adj(params)
    } -> ConvertibleTo<Eigen::Matrix<typename T::Scalar, T::kDof, T::kDof>>;

  // group operations
  {
    T::multiplication(params, params)
    } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kNumParams>>;

  {
    T::inverse(params)
    } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kNumParams>>;

  // Point actions

  {
    T::action(params, point)
    } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kPointDim>>;

  {
    T::toAmbient(point)
    } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kAmbientDim>>;

  // Matrices

  {
    T::compactMatrix(params)
    } -> ConvertibleTo<
        Eigen::Matrix<typename T::Scalar, T::kPointDim, T::kAmbientDim>>;

  {
    T::matrix(params)
    } -> ConvertibleTo<
        Eigen::Matrix<typename T::Scalar, T::kAmbientDim, T::kAmbientDim>>;

  // for tests
  {
    T::exampleTangents()
    } -> ConvertibleTo<std::vector<Eigen::Vector<typename T::Scalar, T::kDof>>>;

  {
    T::exampleParams()
    } -> ConvertibleTo<
        std::vector<Eigen::Vector<typename T::Scalar, T::kNumParams>>>;
};

// Ideally, the LieSubgroupImplConcept is not necessary and all these properties
// can be deduced.
template <class T>
concept LieSubgroupImplConcept = LieGroupImplConcept<T> && requires(
    T g,
    Eigen::Vector<typename T::Scalar, T::kDof> tangent,
    Eigen::Vector<typename T::Scalar, T::kNumParams> params,
    Eigen::Vector<typename T::Scalar, T::kPointDim> point) {
  {
    T::matV(tangent)
    } -> ConvertibleTo<
        Eigen::Matrix<typename T::Scalar, T::kPointDim, T::kPointDim>>;

  {
    T::matVInverse(tangent)
    } -> ConvertibleTo<
        Eigen::Matrix<typename T::Scalar, T::kPointDim, T::kPointDim>>;

  {
    T::topRightAdj(params, point)
    }
    -> ConvertibleTo<Eigen::Matrix<typename T::Scalar, T::kPointDim, T::kDof>>;
};

}  // namespace lie
}  // namespace sophus
