./bazel-bin/tensorflow/compiler/aot/tfcompile --graph=$PWD/tensorflow/compiler/aot/tests/test_graph_tfmatmul.pb \
--config=$PWD/tensorflow/compiler/aot/tests/test_graph_tfmatmul.config.pbtxt --cpp_class="foo::bar::MatMull"
