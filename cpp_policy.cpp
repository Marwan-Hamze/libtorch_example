#include <iostream>
#include <torch/script.h> // One-stop header.

// int main() {

//     try {

//     // Load the model saved with torch.jit. Loading works
//     torch::jit::script::Module module = torch::jit::load("/home/yoshidalab/devel/src/simplecode/libtorch_example/simple_model.pt");
//     std::cout << "Policy with Torch Jit Loaded!" << std::endl;
    
//     // Create input tensor
//     std::vector<torch::jit::IValue> inputs;
//     inputs.push_back(torch::randn({1, 3}));

//     // Run inference
//     at::Tensor output = module.forward(inputs).toTensor();
//     std::cout << "Output: " << output << std::endl;

//         }

//     catch (const c10::Error& e) {
//         std::cerr << "Error loading the model saved with torch.jit.\n";
//         return -1;
//     }

//     try {

//     // Load the model saved with torch.save. The model can't be loaded, and the function returns an error.
//         torch::jit::script::Module test = torch::jit::load("/home/yoshidalab/devel/src/simplecode/libtorch_example/torch_save_simple_model.pt");
//         std::cout << "Policy with Torch Save Loaded!" << std::endl;
//             }

//     catch (const c10::Error& e) {
//         std::cerr << "Error loading the model saved with torch.save.\n";
//         return -1;
//     }

//     return 0;
// }

int main() {

    // Load the actor and critic saved in Python using RL
    torch::jit::script::Module actor = torch::jit::load("/home/yoshidalab/devel/src/simplecode/libtorch_example/kaleido_standing_actor.pt");
    torch::jit::script::Module critic = torch::jit::load("/home/yoshidalab/devel/src/simplecode/libtorch_example/kaleido_standing_critic.pt");
   
    // Create the observation vector
    std::vector<float> obs_data = {
        1, 0, 0, 0,  // Quaternion
        0, 0, 0,     // Angular velocity
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, // Joint positions
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,             // Joint velocities
        0.5, -0.85,                      // Clock
        1.0, 0.0, 0.0,                   // Mode
        0.0, 0.0, 0.0                    // Modref velocities
    };

    // Add random noise between -0.02 and 0.02
    // for (float& val : obs_data) {
    //     val += ((float)rand() / RAND_MAX) * 0.04f - 0.02f;  // Uniform(-0.02, 0.02)
    // }

    // Convert to a torch tensor
    torch::Tensor observation = torch::tensor(obs_data, torch::dtype(torch::kFloat));
    
    // Putting the models in eval mode
    actor.eval();
    critic.eval();

    // Disable gradient tracking
    torch::NoGradGuard no_grad;

    // Run inference
    torch::Tensor actions = actor.forward({observation}).toTensor();
    torch::Tensor value = critic.forward({observation}).toTensor();

    std::cout << "Actions:\n" << actions << std::endl;
    std::cout << "Value:\n" << value << std::endl;

    return 0;
}
