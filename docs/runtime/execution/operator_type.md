# Operator Type

## Purpose

OperatorType uniquely identifies a mathematical operation inside the Runtime.

Unlike Operator implementations, OperatorType contains no execution logic.

It serves only as a lightweight identifier for runtime dispatch.

---

## Responsibilities

OperatorType is responsible for

- identifying an operator,
- serving as part of the dispatch key,
- remaining backend-independent.

---

## Design

OperatorType is represented as an enumeration.

Each mathematical operation owns exactly one OperatorType.

Examples include

- Add
- MatMul
- Relu

Different backend implementations share the same OperatorType.

For example,

```plain text
CPUAddKernel

CUDAAddKernel

AVXAddKernel
```

all correspond to

```plain text
OperatorType::Add
```

---

## Public Interface

OperatorType should be

- lightweight
- constexpr-friendly
- hashable

---

## Future Extensions

Future versions may introduce

- Conv2D
- LayerNorm
- Softmax
- Attention
- GELU

Adding new operators should not affect existing Runtime components.
