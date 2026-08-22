# Mini AI Runtime Architecture

> Version: 0.1
>
> Last Updated: 2026-07-17

---

## 1. Motivation

Modern AI systems rely on sophisticated runtime infrastructures that
efficiently execute tensor computations on modern hardware.

Although frameworks such as PyTorch and TensorFlow provide convenient APIs,
their internal implementations are extremely large and difficult to study.

The goal of this project is **not** to reproduce an industrial framework.

Instead, the objective is to build a lightweight AI runtime from first
principles in order to understand:

- how tensor data is represented,
- how computations are organized,
- how execution is scheduled,
- and how performance is optimized.

This project emphasizes runtime systems rather than machine learning models.

---

## 2. Project Scope

Version 0.1 focuses only on the minimal components required for a runtime.

### Supported

- CPU execution
- Dense tensors
- float32
- Static shapes
- Forward execution
- Single process
- Single machine

### Not Supported

- CUDA
- Automatic differentiation
- Dynamic graph
- Distributed execution
- Mixed precision
- Sparse tensors
- Quantization
- Model serialization

The project intentionally limits its scope in order to keep the architecture
simple and extensible.

---

## 3. Design Philosophy

The entire runtime follows four design principles.

### 3.1 Separation of Responsibilities

Every component should solve exactly one problem.

For example:

- Tensor represents multidimensional data.
- Storage manages memory.
- Operator performs computation.
- Executor schedules execution.

Responsibilities should never overlap.

---

### 3.2 Layered Abstraction

Higher-level components should never directly manipulate lower-level
implementation details.

Each layer communicates only with the layer immediately below it.

This separation allows future extensions without changing existing interfaces.

---

### 3.3 Extensibility

The architecture should support future features such as:

- CUDA backend
- Thread pool
- Memory pool
- Graph optimization
- Operator fusion

without redesigning the core abstractions.

---

### 3.4 Performance Awareness

Correctness always comes first.

Performance optimization should never compromise abstraction.

Optimizations should be introduced incrementally after correctness has been
verified.

---

## 4. System Architecture

The runtime is composed of several independent abstraction layers.

>Application
>
>↓
>
>Executor
>
>↓
>
>Graph
>
>↓
>
>Operator
>
>↓
>
>Tensor
>
>↓
>
>Storage
>
>↓
>
>Allocator

Each layer depends only on lower-level abstractions.

---

### Application

The user-facing program.

Responsible for constructing tensor computations.

---

### Executor

Responsible for executing computation graphs.

The executor does not perform mathematical computations itself; its
responsibility is scheduling and orchestrating the execution of operators.

---

### Graph

Represents the dependency relationships between operators.

The graph contains no mathematical implementation — it encodes only the
structure of the computation.

---

### Operator

Implements mathematical computations.

Examples include:

- Add
- MatMul
- ReLU
- Softmax

Operators consume input tensors and produce output tensors.

---

### Tensor

Represents multidimensional numerical data.

A tensor describes:

- shape
- stride
- data type

A tensor does not own raw memory directly; it references a Storage object.

---

### Storage

Owns a contiguous block of memory.

Storage has no knowledge of tensor dimensions. Its only responsibility is
memory ownership and lifetime management.

---

### Allocator

Responsible for allocating and releasing raw memory.

Future versions may replace the default allocator with specialized
implementations, such as a memory pool or a CUDA allocator, without modifying
higher-level components.

---

## 5. Ownership Model

The runtime separates ownership from representation.

>Allocator owns raw memory allocation.
>
>↓
>
>Storage owns allocated memory.
>
>↓
>
>Tensor references Storage.
>
>↓
>
>Operator references Tensor.
>
>↓
>
>Graph owns Operators.
>
>↓
>
>Executor executes Graph.

This ownership model prevents duplicated memory-management responsibilities.

---

## 6. Runtime Execution

The execution of a tensor program follows the pipeline below.

>User Program
>
>↓
>
>Create Tensors
>
>↓
>
>Create Operators
>
>↓
>
>Construct Computation Graph
>
>↓
>
>Executor::Run()
>
>↓
>
>Traverse Graph
>
>↓
>
>Execute Operators
>
>↓
>
>Produce Output

During execution:

- Operators access Tensor objects.
- Tensor objects access Storage.
- Storage provides access to allocated memory.
- Only the Allocator interacts with the operating system.

---

## 7. Core Design Decisions

This runtime intentionally separates several concepts that are often conflated
in other systems.

### Tensor Is Not Memory

Tensor represents a multidimensional view.

Storage owns memory.

This separation enables:

- tensor views
- slicing
- reshaping
- shared storage

without copying data.

---

### Operator Is Stateless

Operators perform computations.

They should not own tensors.

They should not manage memory.

This simplifies execution scheduling and enables future parallel execution.

---

### Graph Captures Structure Only

The graph describes dataflow dependencies.

It does not perform execution.

Execution belongs exclusively to the Executor.

---

### Executor Controls Execution

Execution policy is independent from graph representation.

Future versions may introduce:

- parallel execution
- asynchronous execution
- CUDA stream execution

without modifying the Graph abstraction.

---

## 8. Future Architecture

Future versions may introduce additional abstractions.

### Backend

- CPU
- CUDA

### Memory

- Memory Pool
- Unified Memory

### Execution

- Thread Pool
- Task Scheduler

### Optimization

- Operator Fusion
- Graph Optimization
- Constant Folding

### Compiler

- Intermediate Representation (IR)
- Graph Lowering

The core abstractions defined in this document should remain unchanged.

---

## 9. Non-Goals

This project is not intended to become:

- a deep learning framework,
- a training library,
- a production inference engine.

Instead, it serves as an educational systems project for understanding the
architecture of modern AI runtimes.

---

## 10. Guiding Question

Every new component added to this project should answer one question:

> How can a high-level tensor program be executed efficiently on modern
> hardware while preserving clean software abstractions?
