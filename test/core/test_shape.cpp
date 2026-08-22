#include <cassert>

#include "runtime/core/shape.h"

int main(){

    runtime::Shape shape({2,3});

    assert(shape.rank()  == 2);
    assert(shape.numel() == 6);
    

}