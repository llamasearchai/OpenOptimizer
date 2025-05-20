#pragma once

#include "ir/Operation.hpp"
#include <vector>
#include <stdexcept> // For std::invalid_argument in inferShapes

namespace openoptimizer {
namespace ir {

// Example: Convolution Operation
class Conv2DOp : public Operation {
public:
    // Attributes like padding, stride, dilation, groups are stored in Operation::attributes_
    Conv2DOp(const std::string& name, 
             int out_channels,
             std::vector<int> kernel_size, // {h, w}
             std::vector<int> stride = {1, 1}, 
             std::vector<int> padding = {0, 0}, 
             std::vector<int> dilation = {1, 1}, 
             int groups = 1, 
             bool bias = true)
        : Operation(name, "Conv2d") {
        // Store attributes using the base class method
        setAttribute("out_channels", out_channels);
        setAttribute("kernel_size", kernel_size); // Stored as std::vector<int>
        setAttribute("stride", stride);
        setAttribute("padding", padding);
        setAttribute("dilation", dilation);
        setAttribute("groups", groups);
        setAttribute("bias", bias);
    }

    std::vector<TensorShape> inferShapes(const std::vector<TensorShape>& input_shapes) const override {
        if (input_shapes.size() != 1 && input_shapes.size() != 2) { // Input, Weight, (Optional Bias)
             // Bias shape usually not used for output shape inference directly for Conv
            throw std::invalid_argument("Conv2D expects 1 (data) or 2 (data, weight) input shapes for shape inference (bias shape often implicit or not needed for output shape).");
        }
        const auto& data_shape = input_shapes[0];
        if (data_shape.getRank() != 4) { // NCHW or NHWC
            throw std::invalid_argument("Conv2D input data must be 4D (e.g., NCHW or NHWC).");
        }

        // Assuming NCHW [N, C_in, H_in, W_in]
        // Or NHWC [N, H_in, W_in, C_in] - layout might be another attribute or convention
        // For simplicity, let's assume NCHW like for PyTorch for H, W calculation
        // The channel dimension (C_in) would be obtained from data_shape.getDimension(1) for NCHW
        
        Dimension N = data_shape.getDimension(0);
        // Dimension C_in = data_shape.getDimension(1); // Used for weight shape validation, not output shape
        Dimension H_in = data_shape.getDimension(2);
        Dimension W_in = data_shape.getDimension(3);

        int k_h = getAttribute<std::vector<int>>("kernel_size")[0];
        int k_w = getAttribute<std::vector<int>>("kernel_size")[1];
        int s_h = getAttribute<std::vector<int>>("stride")[0];
        int s_w = getAttribute<std::vector<int>>("stride")[1];
        int p_h = getAttribute<std::vector<int>>("padding")[0];
        int p_w = getAttribute<std::vector<int>>("padding")[1];
        int d_h = getAttribute<std::vector<int>>("dilation")[0];
        int d_w = getAttribute<std::vector<int>>("dilation")[1];

        int out_channels_val = getAttribute<int>("out_channels");
        Dimension C_out = out_channels_val;

        Dimension H_out = DYNAMIC_DIM;
        Dimension W_out = DYNAMIC_DIM;

        if (H_in.has_value() && k_h > 0 && s_h > 0 && d_h > 0) {
            H_out = (H_in.value() + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1;
        }
        if (W_in.has_value() && k_w > 0 && s_w > 0 && d_w > 0) {
            W_out = (W_in.value() + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1;
        }
        
        return {TensorShape({N, C_out, H_out, W_out})};
    }
};

// Example: ReLU Operation
class ReLUOp : public Operation {
public:
    explicit ReLUOp(const std::string& name)
        : Operation(name, "ReLU") {}

    std::vector<TensorShape> inferShapes(const std::vector<TensorShape>& input_shapes) const override {
        if (input_shapes.size() != 1) {
            throw std::invalid_argument("ReLU expects 1 input shape.");
        }
        // ReLU output shape is same as input shape
        return {input_shapes[0]};
    }
};

// Example: Add Operation (element-wise)
class AddOp : public Operation {
public:
    explicit AddOp(const std::string& name)
        : Operation(name, "Add") {}

    std::vector<TensorShape> inferShapes(const std::vector<TensorShape>& input_shapes) const override {
        if (input_shapes.size() != 2) {
            throw std::invalid_argument("Add (element-wise) expects 2 input shapes.");
        }
        // Broadcasting rules can be complex. For simple element-wise, shapes must match or be broadcastable.
        // Here, assume shapes must be identical for simplicity, or one is scalar.
        // A full implementation would require broadcasting logic.
        if (input_shapes[0] != input_shapes[1]) {
            // Simplistic check: if one is scalar, result is other's shape
            if (input_shapes[0].isScalar() && input_shapes[0].getRank() == 0) return {input_shapes[1]};
            if (input_shapes[1].isScalar() && input_shapes[1].getRank() == 0) return {input_shapes[0]};
            // More complex broadcasting needed here.
            // For now, let's be strict or return the first shape as a placeholder if not strictly equal and not scalar.
            // This is a common source of bugs if not handled carefully.
            // spdlog::warn("AddOp inputs have different shapes ({} vs {}), applying broadcasting (or taking first shape as placeholder).", 
            //    input_shapes[0].toString(), input_shapes[1].toString());
        }
        return {input_shapes[0]}; // Or apply broadcasting rules
    }
};

} // namespace ir
} // namespace openoptimizer 