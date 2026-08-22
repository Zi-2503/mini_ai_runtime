#include "runtime/kernel/cpu/cpu_scale_kernel.h"

namespace runtime{

    template class CPUScaleKernel<float>;
    template class CPUScaleKernel<double>;

}