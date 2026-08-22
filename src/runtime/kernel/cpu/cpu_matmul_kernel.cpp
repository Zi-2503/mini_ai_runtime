#include "runtime/kernel/cpu/cpu_softmax_kernel.h"

namespace runtime{

    template class CPUSoftmaxKernel<float>;
    template class CPUSoftmaxKernel<double>;
    template class CPUSoftmaxKernel<int32_t>;
    template class CPUSoftmaxKernel<int64_t>;

}
