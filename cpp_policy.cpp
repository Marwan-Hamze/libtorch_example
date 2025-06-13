#include <iostream>
#include <torch/script.h> // One-stop header.

int main() {

    try {

    // Load the model saved with torch.jit. Loading works
    torch::jit::script::Module module = torch::jit::load("/home/yoshidalab/devel/src/simplecode/libtorch_example/simple_model.pt");
    std::cout << "Policy with Torch Jit Loaded!" << std::endl;
    
    // Create input tensor
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 3}));

    // Run inference
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << "Output: " << output << std::endl;

        }

    catch (const c10::Error& e) {
        std::cerr << "Error loading the model saved with torch.jit.\n";
        return -1;
    }

    try {

    // Load the model saved with torch.save. The model can't be loaded, and the function returns an error.
        torch::jit::script::Module test = torch::jit::load("/home/yoshidalab/devel/src/simplecode/libtorch_example/torch_save_simple_model.pt");
        std::cout << "Policy with Torch Save Loaded!" << std::endl;
            }

    catch (const c10::Error& e) {
        std::cerr << "Error loading the model saved with torch.save.\n";
        return -1;
    }

    return 0;
}


// int main() {
//     std::cout << "SANITY TEST" << std::endl;
//     torch::Tensor t = torch::rand({2, 3});
//     std::cout << t << std::endl;
//     return 0;
// }
