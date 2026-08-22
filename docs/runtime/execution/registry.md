# Kernel Registry

## Purpose

The Kernel Registry maintains mappings between dispatch requests and concrete
Kernel implementations.

Instead of hardcoding backend selection, the Runtime queries the Registry to
retrieve the appropriate Kernel.

---

## Responsibilities

The Registry is responsible for

- registering Kernels,
- storing dispatch mappings,
- retrieving Kernels.

The Registry performs no computation.

---

## Mapping

Version 1 uses

```plain text
DispatchKey
```

to identify a Kernel.

Example

```plain text
(Add, CPU)
        │
        ▼
CPUAddKernel

(Add, CUDA)
        │
        ▼
CUDAAddKernel
```

---

## Public Interface

The Registry should support

- Register(...)
- Find(...)

No other public APIs are required.

---

## Design Principles

The Registry is a lookup table.

It should remain independent of Tensor, Operator implementations, and backend
details.

---

## Future Extensions

Future versions may support

- dynamic backend loading,
- plugin registration,
- multiple Kernel candidates,
- priority-based selection.
