# Device

## Purpose

A Device represents the execution target of a Tensor or Kernel.

It abstracts the underlying hardware and allows the Runtime to support
multiple execution backends through a unified interface.

A Device does not perform computation.

It only describes where computation and memory reside.

---

## Responsibilities

A Device is responsible for

- identifying the execution backend,
- distinguishing multiple devices of the same backend,
- providing runtime device information.

A Device is NOT responsible for

- memory allocation,
- numerical computation,
- kernel dispatch.

---

## Design

The Runtime separates the backend type from the device instance.

For example,

```plain text
CPU

CUDA:0

CUDA:1
```

all share the same interface.

---

## Public Interface

The Device class should support

- construction
- querying backend type
- querying device index
- comparison
- string conversion

---

## Future Extensions

Future versions may support

- HIP
- Metal
- Vulkan
- OpenCL
- Remote Devices

No API changes should be required when adding new backend types.
