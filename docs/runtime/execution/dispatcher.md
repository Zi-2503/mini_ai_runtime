# Dispatcher

## Purpose

The Dispatcher selects the most appropriate Kernel according to runtime
information.

It acts as the bridge between Operators and backend implementations.

---

## Responsibilities

The Dispatcher decides

- CPU or CUDA
- Float32 or Float64
- different optimized kernels

The Dispatcher itself performs no numerical computation.

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

---

## Dispatch Information

Typical dispatch keys include

- Device
- DataType
- Layout
- Backend

Additional keys may be introduced in future versions.

---

## What the Dispatcher Should Not Do

The Dispatcher should not

- modify tensor data
- implement algorithms
- allocate memory

Its only responsibility is selecting an implementation.

---

## Future Extensions

Future versions may support

- Kernel Registry
- Dynamic Dispatch
- Backend Plugins
