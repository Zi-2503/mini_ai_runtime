#include "runtime/kernel/cpu/cpu_add_kernel.h"

namespace runtime{

    template class CPUAddKernel<float>;
    template class CPUAddKernel<double>;
    template class CPUAddKernel<int32_t>;
    template class CPUAddKernel<int64_t>;

}