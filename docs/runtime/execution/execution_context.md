# Execution Context

## Purpose

ExecutionContext encapsulates all information required by a Kernel to perform a
single execution.

Instead of exposing operator-specific function signatures, the Runtime passes a
single ExecutionContext to every Kernel.

This design keeps the execution interface stable as new operators are added.

---

## Responsibilities

ExecutionContext is responsible for

- storing input tensors,
- storing output tensors,
- providing a unified execution interface.

ExecutionContext performs no numerical computation.

---

## Design

Version 1 contains only Tensor references.

```plain text
ExecutionContext

├── Inputs

└── Outputs
```

Future versions may additionally contain

- operator attributes,
- temporary workspace,
- execution stream,
- profiling information,
- thread pool,
- execution flags.

None of these extensions require changing the Kernel interface.

---

## Public Interface

ExecutionContext should support

- querying input tensors,
- querying output tensors,
- querying input/output counts.

ExecutionContext owns no Tensor memory.

It only references existing Tensor objects.

---

## Design Principles

ExecutionContext is a lightweight execution descriptor.

It should not allocate memory.

It should not perform backend dispatch.

It should not contain mathematical logic.

Its only responsibility is transporting execution data between the Runtime and
the Kernel.

---

## Future Evolution

Future versions may include

```plain text
ExecutionContext

├── Inputs

├── Outputs

├── Attributes

├── Workspace

├── Stream

├── Profiler

└── Execution Flags
```

without changing Kernel or Dispatcher interfaces.
