# Tensor

---

## Purpose

Tensor is the primary data object of the runtime.

A tensor represents a typed, multidimensional view over a block of storage.

Tensor itself does not own memory.

Instead, it references a Storage object while providing metadata that describes
how the underlying bytes should be interpreted.

---

## Motivation

Raw memory alone has no semantic meaning.

For example,

+----------------------------+
| 24 Bytes                   |
+----------------------------+

Without additional information, it is impossible to determine

- how many elements exist,
- what scalar type they represent,
- how they are arranged.

Tensor provides this interpretation by combining

- Shape,
- DataType,
- Storage.

---

## Responsibilities

Tensor is responsible for

- describing tensor shape,
- describing tensor element type,
- referencing underlying storage,
- providing typed memory access.

---

## Non-Responsibilities

Tensor is not responsible for

- allocating memory,
- releasing memory,
- graph execution,
- numerical kernels,
- operator scheduling.

---

## Relationship

           Shape
             │
             │
Storage ─── Tensor ─── DataType

Tensor combines metadata with storage to form a complete runtime object.

---

## Ownership

Tensor does not own raw memory.

Instead,

>Tensor
>↓
>shared_ptr<Storage> 
>↓
>Raw Memory

Multiple tensors may safely reference the same Storage object.

---

## Design Decisions

### Tensor is a lightweight handle

Copying a tensor should not duplicate memory.

Instead, multiple tensors share the same Storage object.

---

### Tensor owns metadata

Tensor exclusively owns

- Shape
- DataType

These metadata describe how Storage should be interpreted.

---

### Tensor provides typed access

Storage stores only raw bytes.

Tensor converts raw memory into typed pointers according to its DataType.

---

## Future Extensions

Future versions may additionally support

- Stride
- Device
- Tensor views
- Slice
- Reshape
- Transpose
- Quantized tensor
