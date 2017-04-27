#pragma once
#include "app/fm_method/proto/fm.pb.h"
namespace PS {
namespace FM {

template <typename V>
class LearningRate {
 public:
  LearningRate(const LearningRateConfig& conf) : conf_(conf) {
    CHECK_GT(alpha(), 0);
    CHECK_GE(beta(), 0);
  }
  ~LearningRate() { }

  V eval(V x = 0) const {
    if (conf_.type() == LearningRateConfig::CONSTANT) {
    // if (x == 0 && beta() == 0) {
      return alpha();
    } else {
      return alpha() / ( x + beta() );
    }
  }

  V alpha() const { return conf_.alpha(); }
  V beta() const { return conf_.beta(); }
 private:
  // const LearningRateConfig& conf() { return conf_; }
  LearningRateConfig conf_;
};


} // namespace FM
} // namespace PS
