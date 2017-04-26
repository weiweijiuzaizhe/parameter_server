#pragma once
#include "util/common.h"
#include "util/matrix.h"
#include "app/fm_method/proto/fm.pb.h"
#include <Eigen/Dense>  //这个是什么语法

namespace PS {
namespace FM {

template<typename T> class Loss {
 public:
  // evaluate the loss value
  virtual T evaluate(const MatrixPtrList<T>& data) = 0;
  // compute the gradients
  virtual void compute(const MatrixPtrList<T>& data, MatrixPtrList<T> gradients) = 0;
};


// scalar loss, that is, a loss which takes as input a real value prediction and
// a real valued label and outputs a non-negative loss value. Examples include
// the hinge hinge loss, binary classification loss, and univariate regression
// loss.
template <typename T>
class ScalarLoss : public Loss<T> {
 public:
  typedef Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > EArray;

  // evaluate the loss value
  virtual T evaluate(const EArray& y, const EArray& Xw) = 0;

  // compute gradients. Skip computing if grad (diag_hessian) is not
  // pre-allocated, namely grad.size() == 0 or diag_hessian.size() == 0
  virtual void compute(const EArray& y, const MatrixPtr<T>& X, const EArray& Xw,
                       EArray gradient, EArray diag_hessian) = 0;

  T evaluate(const MatrixPtrList<T>& data) {
    CHECK_EQ(data.size(), 2);
    SArray<T> y(data[0]->value());
    SArray<T> Xw(data[1]->value());
    CHECK_EQ(y.size(), Xw.size());
    return evaluate(y.EigenArray(), Xw.EigenArray()); //调用的是105行evaluate
  }

  void compute(const MatrixPtrList<T>& data, MatrixPtrList<T> gradients) {  //src/util/matrix.h:14:template<typename V> using MatrixPtrList = std::vector<MatrixPtr<V>>;
    if (gradients.size() == 0) return;

    CHECK_EQ(data.size(), 3);
    auto y = data[0]->value();
    auto X = data[1];
    auto Xw = data[2]->value();

    CHECK_EQ(y.size(), Xw.size());
    CHECK_EQ(y.size(), X->rows());

    CHECK(gradients[0]);
    auto gradient = gradients[0]->value();
    auto  diag_hessian =
        gradients.size()>1 && gradients[1] ? gradients[1]->value() : SArray<T>();
    if (gradient.size() != 0) CHECK_EQ(gradient.size(), X->cols());
    if (diag_hessian.size() != 0) CHECK_EQ(diag_hessian.size(), X->cols());

    if (!y.size()) return;
    compute(y.EigenArray(), X, Xw.EigenArray(), gradient.EigenArray(), diag_hessian.EigenArray());
  }

};

// label = 1 or -1
template <typename T>
class BinaryClassificationLoss : public ScalarLoss<T> {  //这是一个空定义
};


template <typename T>
class LogitLoss : public BinaryClassificationLoss<T> {

 public:
  typedef Eigen::Array<T, Eigen::Dynamic, 1> EArray;
  typedef Eigen::Map<EArray> EArrayMap;

  T evaluate(const EArrayMap& y, const EArrayMap& Xw) {
    return log( 1 + exp( -y * Xw )).sum();
  }

  void compute(const EArrayMap& y, const MatrixPtr<T>& X, const EArrayMap& Xw,
               EArrayMap gradient, EArrayMap diag_hessian) {
    // Do not use "auto tau = ...". It will return an expression and slow down
    // the performace.
    EArray tau = 1 / ( 1 + exp( y * Xw ));

    //有下代码可见要么用一阶的,要么用二阶的值
    if (gradient.size()) 
      gradient = X->transTimes( -y * tau );

    if (diag_hessian.size())
      diag_hessian = X->dotTimes(X)->transTimes( tau * ( 1 - tau ));
  }
};

template <typename T>
class SquareHingeLoss : public BinaryClassificationLoss<T> {  //The Hinge Loss 定义为 E(z) = max(0,1-z),Square Hinge Loss 定义为 E(z) = (max(0,1-z))^2
 public:
  typedef Eigen::Array<T, Eigen::Dynamic, 1> EArray;
  typedef Eigen::Map<EArray> EArrayMap;

  T evaluate(const EArrayMap& y, const EArrayMap& Xw) {
    return (1- y * Xw).max(EArray::Zero(y.size())).square().sum();
  }

  void compute(const EArrayMap& y, const MatrixPtr<T>& X, const EArrayMap& Xw,
               EArrayMap gradient, EArrayMap diag_hessian) {
    gradient = - 2 *  X->transTimes(y * (y * Xw > 1.0).template cast<T>());
  }
};


template <typename T>
class FmSquareLoss : public BinaryClassificationLoss<T> {
 public:
  typedef Eigen::Array<T, Eigen::Dynamic, 1> EArray;  //Eigen是有关线性代数（矩阵、向量等）的c++模板库,Array<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
  typedef Eigen::Map<EArray> EArrayMap;

  T evaluate(const EArrayMap& y, const EArrayMap& Xw) { //本意 Xw 就是X(n*m矩阵)和w(m*1) 的乘积  ,但是这里要让Xw 变为FM的预测值 
   
       return   ( y - Xw ).square().sum();
     //return (1- y * Xw).max(EArray::Zero(y.size())).square().sum();
  }


  void compute(const EArrayMap& y, const MatrixPtr<T>& X, const EArrayMap& Xw,
               EArrayMap gradient, EArrayMap diag_hessian) {
    //TODO

    gradient = - 2 *  X->transTimes(y * (y * Xw > 1.0).template cast<T>());
  }
};





template<typename T>
using LossPtr = std::shared_ptr<Loss<T>>;

template<typename T>
static LossPtr<T> createLoss(const LossConfig& config) {  //这个LossConfig是在proto中定义的
  switch (config.type()) {
    case LossConfig::LOGIT:
      return LossPtr<T>(new LogitLoss<T>());
    case LossConfig::SQUARE_HINGE:
      return LossPtr<T>(new SquareHingeLoss<T>());

    case LossConfig::SQUARE:  //每个loss都要对应一个evaluate,计算当前样本的loss,对应一个compute,计算当前样本的梯度
      return LossPtr<T>(new FmSquareLoss<T>());

    default:
      CHECK(false) << "unknown type: " << config.DebugString();
  }
  return LossPtr<T>(nullptr);
}

} // namespace FM
} // namespace PS
