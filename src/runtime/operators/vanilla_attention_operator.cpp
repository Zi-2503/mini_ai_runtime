#include "runtime/operators/vanilla_attention_operator.h"

#include <cmath>
#include <iostream>

#include "runtime/execution/execution_context.h"
#include "runtime/operators/matmul_operator.h"
#include "runtime/operators/scale_operator.h"
#include "runtime/operators/softmax_operator.h"

namespace runtime {

    AttentionOperator::AttentionOperator(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O)
        : Q_(Q),
          K_(K),
          V_(V),
          O_(O)
    {}

    OperatorType AttentionOperator::type() const noexcept {
        return OperatorType::Attention;
    }

    Status AttentionOperator::Execute(Dispatcher& dispatcher) {

        const Tensor& Q = Q_.get();
        const Tensor& K = K_.get();
        const Tensor& V = V_.get();
        Tensor& O = O_.get();


        if (Q.empty() || K.empty() || V.empty() || O.empty()) {
            return Status::Error(StatusCode::InvalidArgument, "AttentionOperator requires valid tensors.");
        }
        if (Q.rank() != 2 || K.rank() != 2 || V.rank() != 2 || O.rank() != 2) {
            return Status::Error(StatusCode::InvalidArgument, "Attention v1 only supports rank-2 tensors.");
        }

        // Q : [M, D]
        // K : [N, D]
        // V : [N, Dv]
        // O : [M, Dv]

        const size_t M  = Q.dim(0);
        const size_t D  = Q.dim(1);

        const size_t N  = K.dim(0);
        const size_t KD = K.dim(1);

        const size_t VN  = V.dim(0);
        const size_t Dv  = V.dim(1);

        if (D != KD)    return Status::Error(StatusCode::InvalidArgument, "Attention requires Q and K to have the same feature dimension.");
        if (N != VN)    return Status::Error(StatusCode::InvalidArgument, "Attention requires K and V to have the same sequence length.");

        if (O.dim(0) != M || O.dim(1) != Dv)    return Status::Error(StatusCode::InvalidArgument, "Attention output has incorrect shape.");
        
        if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype() || Q.dtype() != O.dtype()) {
            return Status::Error(StatusCode::InvalidArgument, "Attention requires all tensors to have the same dtype.");
        }
        if (Q.device() != K.device() || Q.device() != V.device() || Q.device() != O.device()) {
            return Status::Error(StatusCode::InvalidArgument, "Attention requires all tensors to be on the same device.");
        }

        if (Q.dtype() == DataType::Bool)        return Status::Error(StatusCode::InvalidArgument, "Attention does not support bool tensors.");

        Tensor Kt(Shape({D, N}), K.dtype(), K.device());
        Tensor scores(Shape({M, N}), Q.dtype(), Q.device());
        Tensor scaled_scores(Shape({M, N}), Q.dtype(), Q.device());
        Tensor probabilities(Shape({M, N}), Q.dtype(), Q.device());

        Kt = K.transpose(0, 1);

        // std::cout << "===== QK =====\n";
        // std::cout << "Q = [" << Q.dim(0) << ", " << Q.dim(1) << "]\n";
        // std::cout << "Kt = [" << Kt.dim(0) << ", " << Kt.dim(1) << "]\n";
        // std::cout << "scores = [" << scores.dim(0) << ", " << scores.dim(1) << "]\n";
        // std::cout << "===== PV =====\n";
        // std::cout << "P = [" << probabilities.dim(0) << ", " << probabilities.dim(1) << "]\n";
        // std::cout << "V = [" << V.dim(0) << ", " << V.dim(1) << "]\n";
        // std::cout << "O = [" << O.dim(0) << ", " << O.dim(1) << "]\n";

        MatMulOperator qk(Q, Kt, scores);
        Status status = qk.Execute(dispatcher);
        if(!status.ok())    return status; 

        float scale = 1.0f / std::sqrt(static_cast<float>(D));
        ScaleOperator scaleop(scores, scaled_scores, scale);
        status = scaleop.Execute(dispatcher);
        if(!status.ok())    return status;

        SoftmaxOperator softmaxop(scaled_scores, probabilities);
        status = softmaxop.Execute(dispatcher);
        if(!status.ok())    return status;

        MatMulOperator pv(probabilities, V, O);
        status = pv.Execute(dispatcher);
        if(!status.ok())    return status;

        return Status::OK();

    }

}