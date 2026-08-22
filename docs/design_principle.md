# Design Principles

## Single Responsibility

Every module should have one clear responsibility.

---

## Capability over Implementation

Public APIs expose capabilities rather than implementation details.

---

## Separation of Policy and Mechanism

The Runtime decides what to execute.

The Kernel decides how to execute it.

---

## Composition over Inheritance

Runtime objects should be composed from independent components whenever
possible.

---

## Backend Independence

High-level APIs should not depend on specific hardware backends.

---

## Progressive Complexity

Do not introduce abstractions before they are required by at least two concrete
use cases.
