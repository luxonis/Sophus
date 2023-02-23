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

namespace sophus {
namespace lie {

template <class TScalar>
class QuaternionNumberImpl {
 public:
  using Scalar = TScalar;

  static auto multiplication(
      Eigen::Vector<Scalar, 4> const& lhs_ivec_real,
      Eigen::Vector<Scalar, 4> const& rhs_ivec_real)
      -> Eigen::Vector<Scalar, 2> {
    // complex multiplication
    return Eigen::Vector<Scalar, 2>(
        lhs_real_imag.x() * rhs_real_imag.x() -
            lhs_real_imag.y() * rhs_real_imag.y(),
        lhs_real_imag.x() * rhs_real_imag.y() +
            lhs_real_imag.y() * rhs_real_imag.x());
  }

  static auto squaredNorm(Eigen::Vector<Scalar, 2> const& real_imag) -> double {
    return real_imag.squaredNorm();
  }
};

// template <class TScalar>
// class Rotation2Impl {
//  public:
//   using Scalar = TScalar;
//   using Complex = ComplexNumberImpl<TScalar>;

//   static int const kDof = 1;
//   static int const kNumParams = 2;
//   static int const kPointDim = 2;
//   static int const kAmbientDim = 2;

//   // constructors and factories

//   static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
//     return Eigen::Vector<Scalar, 2>(1.0, 0.0);
//   }

//   static auto areParamsValid(
//       Eigen::Vector<Scalar, kNumParams> const& unit_complex)
//       -> sophus::Expected<Success> {
//     static const Scalar kThr = kEpsilon<Scalar>;
//     const Scalar squared_norm = unit_complex.squaredNorm();
//     using std::abs;
//     if (!(abs(squared_norm - 1.0) <= kThr)) {
//       return FARM_UNEXPECTED(
//           "complex number ({}, {}) is not unit length.\n"
//           "Squared norm: {}, thr: {}",
//           unit_complex[0],
//           unit_complex[1],
//           squared_norm,
//           kThr);
//     }
//     return sophus::Expected<Success>{};
//   }

//   // Manifold / Lie Group concepts

//   static auto exp(Eigen::Vector<Scalar, kDof> const& angle)
//       -> Eigen::Vector<Scalar, kNumParams> {
//     using std::cos;
//     using std::sin;
//     return Eigen::Vector<Scalar, 2>(cos(angle[0]), sin(angle[0]));
//   }

//   static auto log(Eigen::Vector<Scalar, kNumParams> const& unit_complex)
//       -> Eigen::Vector<Scalar, kDof> {
//     using std::atan2;
//     return Eigen::Vector<Scalar, 1>{atan2(unit_complex.y(),
//     unit_complex.x())};
//   }

//   static auto hat(Eigen::Vector<Scalar, kDof> const& angle)
//       -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
//     return Eigen::Matrix<Scalar, 2, 2>{{0, -angle[0]}, {angle[0], 0}};
//   }

//   static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
//       -> Eigen::Matrix<Scalar, kDof, 1> {
//     return Eigen::Matrix<Scalar, kDof, 1>{mat(1, 0)};
//   }

//   static auto adj(Eigen::Vector<Scalar, kNumParams> const&)
//       -> Eigen::Matrix<Scalar, kDof, kDof> {
//     return Eigen::Matrix<Scalar, 1, 1>::Identity();
//   }

//   // group operations

//   static auto inverse(Eigen::Vector<Scalar, kNumParams> const& unit_complex)
//       -> Eigen::Vector<Scalar, kNumParams> {
//     return Eigen::Vector<Scalar, kNumParams>(
//         unit_complex.x(), -unit_complex.y());
//   }

//   static auto multiplication(
//       Eigen::Vector<Scalar, kNumParams> const& lhs_params,
//       Eigen::Vector<Scalar, kNumParams> const& rhs_params)
//       -> Eigen::Vector<Scalar, kNumParams> {
//     auto result = Complex::multiplication(lhs_params, rhs_params);
//     Scalar const squared_norm = result.squaredNorm();

//     // We can assume that the squared-norm is close to 1 since we deal with a
//     // unit complex number. Due to numerical precision issues, there might
//     // be a small drift after pose concatenation. Hence, we need to
//     renormalizes
//     // the complex number here.
//     // Since squared-norm is close to 1, we do not need to calculate the
//     costly
//     // square-root, but can use an approximation around 1 (see
//     // http://stackoverflow.com/a/12934750 for details).
//     if (squared_norm != 1.0) {
//       Scalar const scale = 2.0 / (1.0 + squared_norm);
//       return scale * result;
//     }
//     return result;
//   }

//   // Point actions
//   static auto action(
//       Eigen::Vector<Scalar, kNumParams> const& unit_complex,
//       Eigen::Vector<Scalar, kPointDim> const& point)
//       -> Eigen::Vector<Scalar, kPointDim> {
//     return Complex::multiplication(unit_complex, point);
//   }

//   static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point)
//       -> Eigen::Vector<Scalar, kAmbientDim> {
//     return point;
//   }

//   // matrices

//   static auto compactMatrix(
//       Eigen::Vector<Scalar, kNumParams> const& unit_complex)
//       -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
//     return Eigen::Matrix<Scalar, 2, 2>{
//         {unit_complex.x(), -unit_complex.y()},
//         {unit_complex.y(), unit_complex.x()}};
//   }

//   static auto matrix(Eigen::Vector<Scalar, kNumParams> const& unit_complex)
//       -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
//     return compactMatrix(unit_complex);
//   }

//   // Sub-group concepts
//   static auto matV(Eigen::Vector<Scalar, kNumParams> const&)
//       -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
//     return Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
//   }

//   static auto matVInverse(Eigen::Vector<Scalar, kNumParams> const&)
//       -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
//     return Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
//   }

//   static auto topRightAdj(
//       Eigen::Vector<Scalar, kNumParams> const&,
//       Eigen::Vector<Scalar, kPointDim> const& point)
//       -> Eigen::Matrix<Scalar, kPointDim, kDof> {
//     return Eigen::Matrix<Scalar, 2, 1>(point[1], -point[0]);
//   }

//   // for tests

//   static auto exampleTangents() -> std::vector<Eigen::Vector<Scalar, kDof>> {
//     return std::vector<Eigen::Vector<Scalar, kDof>>({
//         Eigen::Vector<Scalar, kDof>{0.0},
//         Eigen::Vector<Scalar, kDof>{0.00001},
//         Eigen::Vector<Scalar, kDof>{1.0},
//         Eigen::Vector<Scalar, kDof>{-1.0},
//         Eigen::Vector<Scalar, kDof>{5.0},
//         Eigen::Vector<Scalar, kDof>{0.5 * kPi<Scalar>},
//         Eigen::Vector<Scalar, kDof>{0.5 * kPi<Scalar> + 0.00001},
//     });
//   }

//   static auto exampleParams()
//       -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
//     return std::vector<Eigen::Vector<Scalar, kNumParams>>({
//         Rotation2Impl::exp(Eigen::Vector<Scalar, kDof>{0.0}),
//         Rotation2Impl::exp(Eigen::Vector<Scalar, kDof>{1.0}),
//         Rotation2Impl::exp(Eigen::Vector<Scalar, kDof>{0.5 * kPi<Scalar>}),
//         Rotation2Impl::exp(Eigen::Vector<Scalar, kDof>{kPi<Scalar>}),
//     });
//   }
// };

}  // namespace lie
}  // namespace sophus
