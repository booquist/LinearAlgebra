#pragma once

#include <cstddef>
#include "MyLinAlg/Matrix.hpp"

namespace myla {

// Common vector aliases (column vectors)
using Vector2d = Matrix<double, 2, 1>;
using Vector2f = Matrix<float, 2, 1>;
using Vector3d = Matrix<double, 3, 1>;
using Vector3f = Matrix<float, 3, 1>;
using Vector4d = Matrix<double, 4, 1>;
using Vector4f = Matrix<float, 4, 1>;

// Common matrices
using Matrix4f = Matrix<float, 4, 4>;
using Matrix4d = Matrix<double, 4, 4>;
using DynamicMatrixXd = Matrix<double, Dynamic, Dynamic>;
using DynamicVectorXd = Matrix<double, Dynamic, 1>;

} // namespace myla