# libtorch_example

This code serves as a comparison between using `torch.save` or `torch.jit.Trace` for saving a model in PyTorch, before loading it in C++.
Loading the model saved with `torch.save` returns an error, while loading the one saved with `torch.jit.Trace` works fine.

Note that for some reason, in the C++ code, I had to put hard-coded full path to the model so that it can be loadable.

First, run the python script to create the models.

Then, create a `build` repository, use `cd build`, and build with `cmake ..`, then `make`, and finally execute the executable `./libtorch_example`
