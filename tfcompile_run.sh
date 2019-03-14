./bazel-bin/tensorflow/compiler/aot/tfcompile --graph=/workspace/tensorflow/tensorflow/compiler/aot/tests/test_graph_tfmatmul.pb \
--config=/workspace/tensorflow/tensorflow/compiler/aot/tests/test_graph_tfmatmul.config.pbtxt --cpp_class="foo::bar::MatMull"
