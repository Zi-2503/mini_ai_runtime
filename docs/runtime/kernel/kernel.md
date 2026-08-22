# Kernel

## Purpose

A Kernel is the executable implementation of an Operator.

Different hardware platforms may provide different Kernel implementations for
the same Operator.

---

## Responsibilities

A Kernel performs numerical computation.

Typical examples include

- CPU implementation
- CUDA implementation
- AVX implementation

A Kernel should focus only on computation.

---

## Relationship

```plain text
Operator

    │

Dispatcher

    │

Kernel

    │

Tensor
```

---

## What a Kernel Should Not Do

A Kernel should not

- decide which backend to use
- infer tensor shapes
- manage runtime scheduling
- perform operator dispatch

Those responsibilities belong to upper layers.

---

## Design Principles

Each Kernel should

- implement one operation
- target one backend
- remain independent from runtime policy

---

## Future Extensions

Kernel implementations may later support

- SIMD
- CUDA
- Tensor Core
- Mixed Precision
- Asynchronous Execution
