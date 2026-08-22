# Execution Layer

## Purpose

The Execution Layer is responsible for transforming a high-level tensor operation
into an executable backend implementation.

Unlike the Tensor Layer, which represents runtime data objects, the Execution
Layer represents runtime behavior.

---

## Execution Pipeline

```plain text
User API
    │
    ▼
Operator
    │
    ▼
Dispatcher
    │
    ▼
Kernel
    │
    ▼
Tensor / Storage
```

The pipeline consists of three major components:

- Operator
- Dispatcher
- Kernel

Each component has a single responsibility.

---

## Responsibilities

### Operator

Describes **what** computation should be performed.

### Dispatcher

Determines **which implementation** should be executed.

### Kernel

Performs the actual numerical computation.

---

## Design Principles

The execution layer follows three principles.

### Separation of Semantics and Implementation

Operators describe mathematical semantics.

Kernels implement backend-specific computation.

---

### Runtime Dispatch

The runtime chooses an implementation according to runtime information,
rather than hardcoding backend-specific logic.

---

### Extensibility

Adding a new backend should not require modifying existing Operators.

---

## Future Extensions

Future versions may include

- Execution Context
- Kernel Registry
- Backend Manager
- Thread Pool
- CUDA Runtime
