# Operator

## Purpose

An Operator represents a mathematical operation.

Examples include

- Add
- MatMul
- Relu
- Softmax

Operators define the semantics of a computation rather than its implementation.

---

## Responsibilities

An Operator is responsible for

- representing one mathematical operation
- providing a user-facing API
- validating operator-specific arguments
- constructing an ExecutionContext
- selecting execution attributes
- requesting execution through the Dispatcher

An Operator is NOT responsible for

- performing numerical computation
- backend-specific optimizations
- memory allocation

---

## Relationship

```plain text
Operator
    │
    ▼
Dispatcher
    │
    ▼
Kernel
```

The Operator delegates execution to the Dispatcher.

---

## What an Operator Should Not Do

An Operator should not

- contain backend-specific code
- launch CUDA kernels
- perform SIMD optimization
- allocate device memory

These responsibilities belong to lower layers.

---

## Future Extensions

Operators may later support

- automatic shape inference
- graph construction
- operator fusion
