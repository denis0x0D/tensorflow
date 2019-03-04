bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
if [ $? -ne 0 ]; then
  echo "Failed to build tensorflow "
  exit $?
fi
rm ./mnt/tensorflow*.whl
sudo pip uninstall tensorflow
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./mnt/
sudo -E pip install ./mnt/tensorflow*.whl
