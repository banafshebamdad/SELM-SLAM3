#include <SuperPoint.hpp>

namespace SuperPointSLAM
{

/**
 * B.B
 * initializes various convolutional layers and registers them as submodules within the SuperPoint module.
*/
SuperPoint::SuperPoint() {
    /* 
        A Module is registered as a submodule to another Module 
        by calling register_module(), typically from within a parent 
        module’s constructor.
    */

    //SHARED ENCODER
    conv1a = register_module("conv1a", Conv2d(Conv2dOptions(1, c1, 3).stride(1).padding(1)));
    conv1b = register_module("conv1b", Conv2d(Conv2dOptions(c1, c1, 3).stride(1).padding(1)));

    conv2a = register_module("conv2a", Conv2d(Conv2dOptions(c1, c2, 3).stride(1).padding(1)));
    conv2b = register_module("conv2b", Conv2d(Conv2dOptions(c2, c2, 3).stride(1).padding(1)));

    conv3a = register_module("conv3a", Conv2d(Conv2dOptions(c2, c3, 3).stride(1).padding(1)));
    conv3b = register_module("conv3b", Conv2d(Conv2dOptions(c3, c3, 3).stride(1).padding(1)));

    conv4a = register_module("conv4a", Conv2d(Conv2dOptions(c3, c4, 3).stride(1).padding(1)));
    conv4b = register_module("conv4b", Conv2d(Conv2dOptions(c4, c4, 3).stride(1).padding(1)));

    //DETECTOR
    convPa = register_module("convPa", Conv2d(Conv2dOptions(c4, c5, 3).stride(1).padding(1)));
    convPb = register_module("convPb", Conv2d(Conv2dOptions(c5, 65, 1).stride(1).padding(0)));

    //DESCRIPTOR
    convDa = register_module("convDa", Conv2d(Conv2dOptions(c4, c5, 3).stride(1).padding(1)));
    convDb = register_module("convDb", Conv2d(Conv2dOptions(c5, d1, 1).stride(1).padding(0)));
}

/**
 * B.B
 * performs the forward pass of the SuperPoint network
 * applies a series of convolutional layers and max-pooling operations to the input tensor x (in shared encoder, detector, and descriptor sections.)
 * The probabilities are computed using softmax, and the descriptors are normalized.
*/
void SuperPoint::forward(torch::Tensor& x, torch::Tensor& Prob, torch::Tensor& Desc) {
    //SHARED ENCODER
    x = relu(conv1a->forward(x));
    x = relu(conv1b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv2a->forward(x));
    x = relu(conv2b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv3a->forward(x));
    x = relu(conv3b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv4a->forward(x));
    x = relu(conv4b->forward(x));

    //DETECTOR 
    // B.B outputs Prob
    auto cPa = relu(convPa->forward(x));
    auto semi = convPb->forward(cPa); // [B, 65, H/8, W/8]

    //DESCRIPTOR
    // B.B outputs desc
    auto cDa = relu(convDa->forward(x));
    auto desc = convDb->forward(cDa); // [B, 256, H/8, W/8]
    auto dn = norm(desc, 2, 1);
    desc = at::div((desc + EPSILON), unsqueeze(dn, 1));

    //DETECTOR - POST PROCESS
    // B.B Prob is post-processed by applying softmax and reshaping it.
    semi = softmax(semi, 1);            // 65개 채널에서 [H/8 * W/8] 개 원소으 총합 1이 되도록 regression. // B.B Regress so that the total of [H/8 * W/8] elements in 65 channels is 1.
    semi = semi.slice(1, 0, 64);        // remove rest_bin
    semi = semi.permute({0, 2, 3, 1});  // [B, H/8, W/8, 64]

    int Hc = semi.size(1);
    int Wc = semi.size(2);
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8}); 
    semi = semi.permute({0, 1, 3, 2, 4});
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8}); // [B, H, W]

    //Return Tensor
    Prob = semi;    // [B, H, W]
    Desc = desc;    // [B, 256, H/8, W/8]
}

// B.B Vectorized Input Version
/**
 * B.B
 * This method is an overloaded version of the forward method, and it takes a single input tensor x. 
 * It performs the same forward pass operations as described in the previous section. 
 * However, instead of returning separate output tensors, it packs them into a vector and returns that vector.
*/
std::vector<torch::Tensor> SuperPoint::forward(torch::Tensor x) {

    // std::cout << "B.B Welcome to SuperPoint::forward ..." << std::endl; // B.B Starting the Viewer message generated after this message
    x = torch::relu(conv1a->forward(x));
    x = torch::relu(conv1b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv2a->forward(x));
    x = torch::relu(conv2b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv3a->forward(x));
    x = torch::relu(conv3b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv4a->forward(x));
    x = torch::relu(conv4b->forward(x));

    auto cPa = torch::relu(convPa->forward(x));
    auto semi = convPb->forward(cPa);  // [B, 65, H/8, W/8]

    auto cDa = torch::relu(convDa->forward(x));
    auto desc = convDb->forward(cDa);  // [B, d1, H/8, W/8]

    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    semi = torch::softmax(semi, 1);
    semi = semi.slice(1, 0, 64);
    semi = semi.permute({0, 2, 3, 1});  // [B, H/8, W/8, 64]

    int Hc = semi.size(1);
    int Wc = semi.size(2);
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8});
    semi = semi.permute({0, 1, 3, 2, 4});
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8});  // [B, H, W]

    std::vector<torch::Tensor> ret;
    ret.push_back(semi);
    ret.push_back(desc);

    return ret;
  }

} // Namespace NAMU_TEST END
