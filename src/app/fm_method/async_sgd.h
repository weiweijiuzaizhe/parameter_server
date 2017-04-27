/**
 * @file   async_sgd.h
 * @brief  Asynchronous stochastic gradient descent to solve linear methods.
 */
#pragma once
#include <random>
#include "ps.h"
#include "learner/sgd.h"
#include "util/evaluation.h"
#include "parameter/kv_vector.h"
#include "parameter/kv_map.h"
#include "app/fm_method/learning_rate.h"
#include "app/fm_method/proto/fm.pb.h"
#include "app/fm_method/loss.h"
#include "app/fm_method/penalty.h"
namespace PS {
namespace FM {

/**
 * @brief The scheduler node
 */
class AsyncSGDScheduler : public ISGDScheduler {  //Scheduler端几乎没有变化
 public:
  AsyncSGDScheduler(const Config& conf)
      : ISGDScheduler(), conf_(conf) {
    Workload load;
    *load.mutable_data() = conf_.training_data();
    load.mutable_data()->set_ignore_feature_group(true);
    load.set_replica(conf_.async_sgd().num_data_pass());
    load.set_shuffle(true);
    workload_pool_ = new WorkloadPool(load);
  }
  virtual ~AsyncSGDScheduler() { }

 private:
  Config conf_;
};

/**
 * @brief A server node
 */
template <typename V>
class AsyncSGDServer : public ISGDCompNode {
 public:
  AsyncSGDServer(const Config& conf): ISGDCompNode(), conf_(conf) {  //异步服务
    SGDState state(conf_.penalty(), conf_.learning_rate());
    state.reporter = &(this->reporter_);
    if (conf_.async_sgd().algo() == SGDConfig::FTRL) {  //algo: FTRL
      LOG(INFO) << "IN AsyncSGDServer: COME TO FTRL WAY ";
      auto model = new KVMap<Key, V, FTRLEntry, SGDState>();
      model->set_state(state);
      model_ = model;
      
    } else {

 //     原来代码里如下,怀疑是adagrad没有完成或者和proto的格式懒得改才这样写的     
 //     if (conf_.async_sgd().ada_grad()) {
 //       model_ = new KVMap<Key, V, AdaGradEntry, SGDState>();  //被SGD调用

     if (conf_.async_sgd().algo() == SGDConfig::STANDARD) {  //在conf里写一下STANDARD,实际调用自己写的sgd
      LOG(INFO) << "IN AsyncSGDServer: COME TO SGD WAY ";
      //auto  model = new KVMap<Key, V, AdaGradEntry, SGDState>(); 
      auto  model = new KVMap<Key, V, SGDEntry, SGDState>();   //typedef uint64 Key;
      model->set_state(state);
      model_ = model;

      } else {
        CHECK(false);
      //   model_ = new KVStore<Key, V, AdaGradEntry<V>, SGDState<V>>();
      }
    }
  }

  virtual ~AsyncSGDServer() {
    delete model_;
  }

  void SaveModel() {
    auto output = conf_.model_output();
    if (output.format() == DataConfig::TEXT) {
      CHECK(output.file_size());
      std::string file = output.file(0) + "_" + MyNodeID();
      CHECK_NOTNULL(model_)->WriteToFile(file);
      LOG(INFO) << MyNodeID() << " written the model to " << file;
    }
  }

  virtual void ProcessRequest(Message* request) {
    if (request->task.sgd().cmd() == SGDCall::SAVE_MODEL) {
      SaveModel();
    }
  }
 protected:
  Parameter* model_ = nullptr;
  Config conf_;

  /**
   * @brief Progress state
   */
  struct SGDState {  //更新SGD的状态
    SGDState() { }
    SGDState(const PenaltyConfig& h_conf, const LearningRateConfig& lr_conf) {
      lr = std::shared_ptr<LearningRate<V>>(new LearningRate<V>(lr_conf));
      h = std::shared_ptr<Penalty<V>>(createPenalty<V>(h_conf));
    }
    virtual ~SGDState() { }

    void Update() {
      if (!reporter) return;
      SGDProgress prog;
      prog.set_nnz(nnz);
      prog.set_weight_sum(weight_sum); weight_sum = 0;
      prog.set_delta_sum(delta_sum); delta_sum = 0;
      reporter->Report(prog);
    }

    void UpdateWeight(V new_weight, V old_weight) {
      // LL << new_weight << " " << old_weight;
      if (new_weight == 0 && old_weight != 0) {
        -- nnz; 
      } else if (new_weight != 0 && old_weight == 0) {
        ++ nnz;
      }
      weight_sum += new_weight * new_weight;
      V delta = new_weight - old_weight;
      delta_sum += delta * delta;
    }

    std::shared_ptr<LearningRate<V>> lr;
    std::shared_ptr<Penalty<V>> h;

    int iter = 0;
    size_t nnz = 0;
    V weight_sum = 0;
    V delta_sum = 0;
    V max_delta = 1.0;  // maximal change of weight
    MonitorSlaver<SGDProgress>* reporter = nullptr;
  };

  /**
   =* @brief An entry for FTRL
   */
  struct FTRLEntry {
    V w = 0;  // not necessary to store w, because it can be computed from z
    V z = 0;
    V sqrt_n = 0;

    void Set(const V* data, void* state) {
      SGDState* st = (SGDState*) state;
      // update model
      V w_old = w;
      V grad = *data;
      V sqrt_n_new = sqrt(sqrt_n * sqrt_n + grad * grad);
      V sigma = (sqrt_n_new - sqrt_n) / st->lr->alpha();
      z += grad  - sigma * w;
      sqrt_n = sqrt_n_new;
      V eta = st->lr->eval(sqrt_n);
      w = st->h->proximal(-z*eta, eta);

      // update status
      st->UpdateWeight(w, w_old);
    }

    void Get(V* data, void* state) { *data = w; }
  };

  /**
   * @brief An entry for adaptive gradient
   */
  struct AdaGradEntry {  //和sgd的不同点在于对于学习率会根据梯度变化,见 http://blog.csdn.net/luo123n/article/details/48239963
    void Set(const V* data, void* state) {
      
      SGDState* st = (SGDState*) state;
      // update model
      V grad = *data;
      sum_sq_grad += grad * grad;
      V eta = st->lr->eval(sqrt(sum_sq_grad));
      V w_old = weight;
      weight = st->h->proximal(weight - eta * grad, eta);

      // update status
      st->UpdateWeight(weight, w_old);
    }

    void Get(V* data, void* state) { *data = weight; }
    V weight = 0;
    V sum_sq_grad = 0;
  };

  // /**
  //  * @brief An entry for standard gradient desecent
  //  */
  struct SGDEntry {  //和sgd的不同点在于对于学习率会根据梯度变化,见 http://blog.csdn.net/luo123n/article/details/48239963
    void Set(const V* data, void* state) {
      
      SGDState* st = (SGDState*) state  ;    // update model;

      V grad = *data;
 //     sum_sq_grad += grad * grad;
 //     V eta = st->lr->eval(sqrt(sum_sq_grad));  //这里调用的是src/app/fm_method/learning_rate.h中的eval,是根据grad计算出一个步长
      V w_old = weight;
      weight = st->h->proximal(weight - 0.01 * grad, 0.01);

      // update status
      st->UpdateWeight(weight, w_old);
    }

    void Get(V* data, void* state) { *data = weight; }
    V weight = 0;
    V sum_sq_grad = 0;
  };
};

/**
 * @brief A worker node
 */
template <typename V>
class AsyncSGDWorker : public ISGDCompNode {
 public:
  AsyncSGDWorker(const Config& conf)
      : ISGDCompNode(), conf_(conf) {
    loss_ = createLoss<V>(conf_.loss());
  }
  virtual ~AsyncSGDWorker() { }

  virtual void ProcessRequest(Message* request) {
    const auto& sgd = request->task.sgd();
    if (sgd.cmd() == SGDCall::UPDATE_MODEL) {
      // do workload
      UpdateModel(sgd.load());

      // reply the scheduler with the finished id
      Task done;
      done.mutable_sgd()->set_cmd(SGDCall::UPDATE_MODEL);
      done.mutable_sgd()->mutable_load()->add_finished(sgd.load().id());
      Reply(request, done);
    }
  }

  virtual void Run() {
    // request workload from the scheduler
    Task task;
    task.mutable_sgd()->set_cmd(SGDCall::REQUEST_WORKLOAD);
    Submit(task, SchedulerID());
  }

 private:
  /**
   * @brief Process a file
   *
   * @param load
   */
  void UpdateModel(const Workload& load) {
    LOG(INFO) << MyNodeID() << ": accept workload " << load.id();
    VLOG(1) << "workload data: " << load.data().ShortDebugString();
    const auto& sgd = conf_.async_sgd();
    MinibatchReader<V> reader;
    reader.InitReader(load.data(), sgd.minibatch(), sgd.data_buf());  //sgd.minibatch()每个batch读取多少条数据,传入的load看起来只在这里用了一次
    reader.InitFilter(sgd.countmin_n(), sgd.countmin_k(), sgd.tail_feature_freq());
    reader.Start();

    processed_batch_ = 0;
    int id = 0;
    SArray<Key> key;

    for (; ; ++id) {
      mu_.lock();
      auto& data = data_[id];
      mu_.unlock();
      
      /**
       pair_.resize(idx.size());  //resize()可以认为是分配空间  ,pair_是一个SArray<Pair> 类型
        for (size_t i = 0; i < idx.size(); ++i) {
        pair_[i].k = idx[i];
        pair_[i].i = i;
        }
        */

      if (!reader.Read(data.first, data.second, key)) break; //data.first是Y, data.second是x,定义在sgd.h

      VLOG(1) << "load minibatch " << id << ", X: "
              << data.second->rows() << "-by-" << data.second->cols();

      // pull the weight
      auto req = Parameter::Request(id, -1, {}, sgd.pull_filter());

      //key p length array contains the original feature id in the data

      //TO  要在这里改变一下,建立一个 max_feature_n ,对应的隐向量的id 为 feature_id + max_feature_n + 1 ,因为typedef uint64 Key,所以位数足够
      // 遍历 SArray<Key> key 里面的值,push_back  对应的feature_id + max_feature_n + 1

      SArray<Key> fm_key;
      Key max_feature_n = 1000000000;  //10亿  
      
      fm_key.resize( key.size() * 2 );
      
      for (size_t i = 0; i < key.size(); ++i) {
          fm_key[i] = key[i];
       }

      for (size_t i = key.size(); i < key.size() * 2; ++i) {  
          fm_key[i] = key[i] +  max_feature_n + 1;
       }


       model_[id].key = fm_key;    //  KVVector<Key, V> model_; 要改的是这行,FM的key比LR的key要多n+1个
       model_.Pull(req, fm_key, [this, id]() { ComputeGradient(id); }); //最后的变量一个匿名的回调函数


      //model_[id].key = key;    //  KVVector<Key, V> model_; 要改的是这行,FM的key比LR的key要多n+1个
      //model_.Pull(req, key, [this, id]() { ComputeGradient(id); });   //pull之后就计算这个点上的梯度,ComputeGradient 中有 改变model的值并且push到server端的动作


  /**
   * @brief Pull data from servers
   *
   * @param request
   * @param keys n keys
   * @param callback called when responses of this request is received
   *
   * @return the timestamp
   */
  
  /*
    int Pull(const Task& request, 
           const SArray<K>& keys,
           const Message::Callback& callback = Message::Callback());

  */

    }

    while (processed_batch_ < id) { usleep(500); }  //这个worker跑得太快了
    LOG(INFO) << MyNodeID() << ": finished workload " << load.id();
  }

  /**
   * @brief Compute gradient
   *
   * @param id minibatch id
   */
  void ComputeGradient(int id) {  //这里计算的梯度是obj的梯度

    mu_.lock();
    auto Y = data_[id].first;  //X,Y的类型定义都是:typedef Eigen::Map<EArray> EArrayMap;
    auto X = data_[id].second;
    data_.erase(id);
    mu_.unlock();

    CHECK_EQ(X->rows(), Y->rows());  //  检查行数一致
    VLOG(1) << "compute gradient for minibatch " << id;

    // evaluate
    SArray<V> Xw(Y->rows());  // Y->rows()返回一个int值, Zero-copy constructor, namely just copy the pointer,生成变量Xw
    
    auto w = model_[id].value;  //  []是重载的运算符,Returns the key-vale pairs in channel "chl",value是双数组的第二个数组,第一个是key,都是SArray<V>类型的,model的大小可能和  fm_key.resize( key.size() * 2 );  有关

    LOG(INFO) << "size of  w:" << w.size() ;



    //Xw 的类型  SArray<V>, EigenArray的定义: EArrayMap EigenArray() const { return EArrayMap(data(), size()); }
    //Xw.EigenArray() = *X * w.EigenArray();  // auto X = data_[id].second,是数据的各个维度值,X是n*m,w是 m*1,Xw是 n*1,根据后面的意思,Xw是预测值

    //


    //在这里要根据X和W的值计算出每个样本的预测值,给接下来的auc,accuracy,loss_->evaluate使用
    for (size_t i = 0; i < Y->rows(); ++i){


    } 





    SGDProgress prog;

    prog.add_objective(loss_->evaluate({Y, Xw.SMatrix()}));  //定义在123行


    // not with penalty. penalty_->evaluate(w.SMatrix());
    prog.add_auc(Evaluation<V>::auc(Y->value(), Xw));  //  Xw是预测值
    prog.add_accuracy(Evaluation<V>::accuracy(Y->value(), Xw));
    prog.set_num_examples_processed(
        prog.num_examples_processed() + Xw.size()
        );
    this->reporter_.Report(prog);

    // compute the gradient
    SArray<V> grad(X->cols());  //是构造函数,产生变量grad,类型SArray<V>
    loss_->compute( {Y, X, Xw.SMatrix()},  {grad.SMatrix()} );  //调用的是44行   //src/util/matrix.h:14:template<typename V> using MatrixPtrList = std::vector<MatrixPtr<V>>;  {}是匿名函数作为生成对应的输入变量,但是在compute(2个参数)内部还是调用了特定loss的(4个参数)

    // push the gradient
    auto req = Parameter::Request(id, -1, {}, conf_.async_sgd().push_filter()); 
    // grad.EigenArray() /= (V)Y->rows();
    // LL << grad;
    model_.Push(req, model_[id].key, {grad}, [this](){ ++ processed_batch_; });
    model_.Clear(id);
  }

private:
  KVVector<Key, V> model_;
/**
  KVVector的核心是有这样一个双数组数据结构来表示一个稀疏向量
  struct KVPairs {
    SArray<K> key;    // [key_0,  ..., key_n]
    SArray<V> value;  // [val_00, ..., val_0k, ..., val_n0, ..., val_nk]
  };
*/

  LossPtr<V> loss_;

  // minibatch_id, Y, X
  std::unordered_map<int, std::pair<MatrixPtr<V>, MatrixPtr<V>>> data_;

  std::mutex mu_;
  std::atomic_int processed_batch_;
  int workload_id_ = -1;

  Config conf_;
};

} // namespace FM
} // namespace PS
