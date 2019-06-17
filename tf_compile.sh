./bazel-bin/tensorflow/compiler/aot/tfcompile --graph=$PWD/tensorflow/compiler/aot/tests/test_graph_tfadd2.pb \
--config=$PWD/tensorflow/compiler/aot/tests/test_graph_tfadd2.config.pbtxt --cpp_class="foo::bar::Add"
