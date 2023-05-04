#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    Eigen::MatrixXd A(2 * count, 4);

    for (int i = 0; i < count; ++i) {
        cv::Matx14d a0 = Ps[i].row(2) * ms[i](0) - Ps[i].row(0) * ms[i](2);
        cv::Matx14d a1 = Ps[i].row(2) * ms[i](1) - Ps[i].row(1) * ms[i](2);

        A.row(2 * i) << a0(0), a0(1), a0(2), a0(3);
        A.row(2 * i + 1) << a1(0), a1(1), a1(2), a1(3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd X = svd.matrixV().col(3);
    X /= X[3];

    return {X[0], X[1], X[2], X[3]};
}
