#include "gtest/gtest.h"
#include "box/vectors-inl.h"

using namespace PS;

TEST(Vectors, xxx) {
  map<int, Header> hs;

  {
  Header x;
  {
    Header h;
    h.mutable_key()->set_empty(1);
    h.mutable_value()->set_empty(1);
    h.mutable_push()->set_delta(1);
    h.mutable_pull()->set_delta(1);
    hs[0] = h;
    x = h;
  }
  }
  hs.erase(0);

}

void MoveData(Vectors<double> *w) {

  w->vec(1) = w->vec(0);
  w->reset_vec(0);
  LL << w->DebugString();
}

TEST(Vectors, Aggregator) {

  int n = 6;  // global #feature
  int m = 4;  // local #feature
  // local gradients
  DVec g1 = DVec::Ones(m);
  DVec g2 = DVec::Ones(m);

  // FLAGS_num_client = 2;
  // FLAGS_num_server = 2;
  // FLAGS_my_type = "s";
  // pid_t pid = fork();
  // if (pid == 0) {
  //   FLAGS_my_type = "c";
  //   pid_t pid2 = fork();
  //   if (pid2 == 0) {
  //     FLAGS_my_type = "c";
  //     FLAGS_my_rank ++;
  //     pid_t pid3 = fork();
  //     if (pid3 == 0)
  //       FLAGS_my_type = "s";
  //   }
  // }
  // // srand(time(0)+my_seed);

  int delay = 2;

  // s0: v v v
  // s1:       v v v
  // c0: v v   v v
  // c1:   v v   v v

  if (FLAGS_my_type == "client") {
    // local keys
    XArray<Key> keys;
    DVec g;
    if (FLAGS_my_rank == 0) {
      keys = XArray<Key>({0,1,3,4});
      g = g1;
    } else {
      keys = XArray<Key>({1,2,4,5});
      g = g2;
    }
    // the first col is weight, the second is gradient
    Vectors<double> w("haha", n, 2, keys);
    w.SetMaxPullDelay(delay);
    w.vec(0) = DVec::Zero(m);

    int max_iter = 10;
    for (int i = 0; i <= max_iter; ++i) {
      w.vec(0) = g * (1<<(max_iter - i));
      w.PushPull(KeyRange::All(), {0}, kValue, {1}, kDelta);
      // LL << w.DebugString();
      double res = w.vec(1).sum();
      double expect_min = (1<<(max_iter+1)) - (1<<(max_iter+1-std::max(0,i-delay)));
      double expect_max = (1<<(max_iter+1)) - (1<<(max_iter+1-i));
      // LL << i << " " << res/6.0 << " " << expect_min << " " << expect_max;
      EXPECT_LE(res/6.0, expect_max);
      EXPECT_GE(res/6.0, expect_min);
    }
    w.Wait();
    EXPECT_EQ(w.vec(1).sum()/6.0, (1<<(max_iter+1))-1);
    std::this_thread::sleep_for(seconds(2));
  } else {
    Vectors<double> w("haha", n, 2);
    w.SetAggregator(NodeGroup::kClients);
    w.SetAggregatorFunc(NewPermanentCallback(MoveData, &w));
    std::this_thread::sleep_for(seconds(10));
  }

  int ret; wait(&ret);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  EXPECT_EQ(RUN_ALL_TESTS(), 0);
  // LL << "exists";
}
