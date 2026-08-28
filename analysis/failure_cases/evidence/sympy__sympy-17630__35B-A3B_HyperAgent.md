# sympy__sympy-17630  (35B-A3B / HyperAgent)

**Failure mode:** symptom-vs-root  
**Note:** lib blockmatrix.py; gold matexpr.py

- base (no Librarian) `hyperagent_seed0`: passed=True
- +Librarian `hyperagent_librarian_seed0`: passed=False

**Gold/base edited files:** ['sympy/matrices/expressions/matexpr.py']

**+Librarian edited files:** ['sympy/matrices/expressions/blockmatrix.py']

## Issue

```
Exception when multiplying BlockMatrix containing ZeroMatrix blocks
When a block matrix with zero blocks is defined

```
>>> from sympy import *
>>> a = MatrixSymbol("a", 2, 2)
>>> z = ZeroMatrix(2, 2)
>>> b = BlockMatrix([[a, z], [z, z]])
```

then block-multiplying it once seems to work fine:

```
>>> block_collapse(b * b)
Matrix([
[a**2, 0],
[0, 0]])
>>> b._blockmul(b)
Matrix([
[a**2, 0],
[0, 0]])
```

but block-multiplying twice throws an exception:

```
>>> block_collapse(b * b * b)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jan/.pyenv/versions/3.7.4/lib/python3.7/site-packages/sympy/matrices/expressions/blockmatrix.py", line 297, in block_collapse
    result = rule(expr)
  File "/home/jan/.pyenv/versions/3.7.4/lib/python3.7/site-packages/sympy/strategies/core.py", line 11, in exhaustive_rl
    new, old = rule(expr), expr
  File "/home/jan/.pyenv/versions/3.7.4/lib/python3.7/site-packages/sympy/strategies/core.py", line 44, in chain_rl
    expr = rule(expr)
  File "/home/jan/.pyenv/versions/3.7.4/lib/python3.7/site-packages/sympy/strategies/core.py", line 11, in exhaustive_rl
    new, old = rule(expr), expr
  File "/home/jan/.pyenv/versions/3.7.4/lib/python3.7/site-packages/sympy/strategies/core.py", line 33, in conditioned_rl
    return rule(expr)
  File "/home/jan/.pyenv/versions/3.7.4/lib/python3.7/site-packages/sympy/strategies/core.py", line 95, in switch_rl
    return rl(expr)
  File "/home/jan/.pyenv/versions/3.7.4/lib/python3.7/site-packages/sympy/matrices/expressions/blockmatri
```

## Librarian submit_result answers (the search guidance)

1. Found `BlockMatrix` class in `/testbed/sympy/matrices/expressions/blockmatrix.py`:
- `colblocksizes` property: lines 153-155
- `_blockmul` method: lines 164-169
- `BlockDiagMatrix._blockmul` (subclass override): lines 331-336

2. Full file `/testbed/sympy/matrices/expressions/blockmatrix.py` (573 lines) was already retrieved. Key sections:
- `BlockMatrix` class: lines 23-272
- `BlockMatrix._blockmul`: lines 164-169
- `BlockDiagMatrix._blockmul`: lines 331-336
- `block_collapse` function: lines 348-398

3. `bc_matmul` function is at `/testbed/sympy/matrices/expressions/blockmatrix.py` lines 454–477.

4. Lines 273-350 cover the `BlockDiagMatrix` subclass (lines 274-346) and the start of `block_collapse` function (line 348). This includes `BlockDiagMatrix.__new__`, `diag`, `blocks`, `shape`, `blockshape`, `rowblocksizes`, `colblocksizes` properties, `_eval_inverse`, `_eval_transpose`, `_blockmul`, and `_blockadd` methods.

5. Current `_blockmul` implementations in `/testbed/sympy/matrices/expressions/blockmatrix.py`:

**BlockMatrix._blockmul** (lines 164-192): Handles multiplication between BlockMatrix objects. Checks if `other` is a BlockMatrix with matching col/row block sizes, performs block multiplication via `self.blocks*other.blocks`, then converts the result back to a BlockMatrix, replacing scalar zeros with ZeroMatrix objects as needed.

**BlockDiagMatrix._blockmul** (lines 356-361): Handles multiplication between BlockDiagMatrix objects. If `other` is also a BlockDiagMatrix with matching block sizes, returns a BlockDiagMatrix with element-wise products. Otherwise, falls back to `BlockMatrix._blockmul`.

## base (PASS) patch

```diff
diff --git a/sympy/matrices/expressions/matexpr.py b/sympy/matrices/expressions/matexpr.py
index fd1e6d970a..220cdbc8ff 100644
--- a/sympy/matrices/expressions/matexpr.py
+++ b/sympy/matrices/expressions/matexpr.py
@@ -627,7 +627,10 @@ def _postprocessor(expr):
                 # manipulate them like non-commutative scalars.
                 return cls._from_args(nonmatrices + [mat_class(*matrices).doit(deep=False)])
 
-        return mat_class(cls._from_args(nonmatrices), *matrices).doit(deep=False)
+        if nonmatrices:
+            return mat_class(cls._from_args(nonmatrices), *matrices).doit(deep=False)
+        else:
+            return mat_class(*matrices).doit(deep=False)
     return _postprocessor
```

## +Librarian (FAIL) patch

```diff
diff --git a/sympy/matrices/expressions/blockmatrix.py b/sympy/matrices/expressions/blockmatrix.py
index 95302a05ca..299b3950f8 100644
--- a/sympy/matrices/expressions/blockmatrix.py
+++ b/sympy/matrices/expressions/blockmatrix.py
@@ -164,7 +164,32 @@ def structurally_equal(self, other):
     def _blockmul(self, other):
         if (isinstance(other, BlockMatrix) and
                 self.colblocksizes == other.rowblocksizes):
-            return BlockMatrix(self.blocks*other.blocks)
+            product = self.blocks*other.blocks
+            # product may be a regular Matrix (ImmutableDenseMatrix) with
+            # scalar elements (like Zero) instead of matrices.
+            # We need to convert it to a list of lists of matrices.
+            if getattr(product, 'is_Matrix', False):
+                # Convert product to list of lists, replacing scalar zeros
+                # with appropriate ZeroMatrix objects
+                prod_shape = product.shape
+                new_blocks = []
+                for i in range(prod_shape[0]):
+                    row = []
+                    for j in range(prod_shape[1]):
+                        elem = product[i, j]
+                        # Check if element is a scalar (not a matrix)
+                        if not getattr(elem, 'is_Matrix', False):
+                            # Replace with ZeroMatrix of appropriate size
+                            # Use the rowblocksize from self and colblocksize
+                            # from other for the zero matrix dimensions
+                            row_size = self.rowblocksizes[i] if i < len(self.rowblocksizes) else 0
+                            col_size = other.colblocksizes[j] if j < len(other.colblocksizes) else 0
+                            row.append(ZeroMatrix(row_size, col_size))
+                        else:
+                            row.append(elem)
+                    new_blocks.append(row)
+                return BlockMatrix(new_blocks)
+            return BlockMatrix(product)
 
         return self * other
```
