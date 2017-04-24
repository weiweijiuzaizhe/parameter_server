#include "ps.h"
#include "app/fm_method/async_sgd.h"
#include "app/fm_method/darlin.h"
#include "app/fm_method/model_evaluation.h"

namespace PS {
App* App::Create(const string& conf_str) {
  using namespace FM;
  // parse config
  Config conf;
  CHECK(google::protobuf::TextFormat::ParseFromString(conf_str, &conf))
      << " failed to parse conf: " << conf.ShortDebugString();

  // create app
  auto my_role = MyNode().role();
  App* app = nullptr;
  if (conf.has_darlin()) {  //这个has_darlin是在fm.pb.h
    
    LOG(INFO) << "come to has_darlin";  //conf中有darlin,是走了这一步,实际上是一个叫块坐标下降( block-coordinate descent )的算法

    if (my_role == Node::SCHEDULER) {
      app = new DarlinScheduler(conf);
    } else if (my_role == Node::WORKER) {
      app = new DarlinWorker(conf);
    } else if (my_role == Node::SERVER) {
      app = new DarlinServer(conf);
    }
  } else if (conf.has_async_sgd()) {  //这个has_async_sgd是在fm.pb.h
    
    LOG(INFO) << "come to has_async_sgd";  //conf中有sgd,是走了这一步

    typedef float Real;
    if (my_role == Node::SCHEDULER) {
      app = new AsyncSGDScheduler(conf);
    } else if (my_role == Node::WORKER) {
      app = new AsyncSGDWorker<Real>(conf);
    } else if (my_role == Node::SERVER) {
      app = new AsyncSGDServer<Real>(conf);
    }
  } else if (conf.has_validation_data()) {
    app = new ModelEvaluation(conf);
  }
  CHECK(app) << "fail to create " << conf.ShortDebugString()
             << " at " << MyNode().ShortDebugString();
  return app;
}
} // namespace PS

int main(int argc, char *argv[]) {
  PS::RunSystem(argc, argv);
  return 0;
}
