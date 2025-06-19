# libtorch_example

### cpp_policy.cpp
The first part of this code serves as a comparison between using `torch.save` or `torch.jit.Trace` for saving a model in PyTorch, before loading it in C++.
Loading the model saved with `torch.save` returns an error, while loading the one saved with `torch.jit.Trace` works fine.

Note that for some reason, in the C++ code, I had to put hard-coded full path to the model so that it can be loadable.

First, run the python script `python_policy.py` to create the models.

The second part of the code (the other `main` function) loads the actor and critic models obtained by training kaleido (humanoid robot) with RL, and performs inference.
This is just to compare the output of inference in C++ to Python's, all while using the same input and models.

Then, create a `build` repository, use `cd build`, and build with `cmake ..`, then `make`, and finally execute the executable `./libtorch_example`

### compare.py

Here, I load the trained policies with reinforcement learning with Kaleido. This file compares the output of the actors and critics saved with using `torch.save` or `torch.jit.Trace`. The outputs of the actors (actions) are totally identical, same for the outputs of the critics (values) 
