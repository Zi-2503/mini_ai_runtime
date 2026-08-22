#include "runtime/kernel/cpu/cpu_softmax_kernel.h"

namespace runtime{

    template class CPUSoftmaxKernel<float>;
    template class CPUSoftmaxKernel<double>;

}