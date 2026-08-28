# pydata__xarray-6938  (35B-A3B / BOAD)

**Failure mode:** adjacent-module  
**Note:** lib dataset.py; gold variable.py

- base (no Librarian) `boad_seed0`: passed=True
- +Librarian `boad_librarian_seed0`: passed=False

**Gold/base edited files:** ['xarray/core/variable.py']

**+Librarian edited files:** ['xarray/core/dataset.py']

## Issue

```
`.swap_dims()` can modify original object
### What happened?

This is kind of a convoluted example, but something I ran into. It appears that in certain cases `.swap_dims()` can modify the original object, here the `.dims` of a data variable that was swapped into being a dimension coordinate variable.

### What did you expect to happen?

I expected it not to modify the original object.

### Minimal Complete Verifiable Example

```Python
import numpy as np
import xarray as xr

nz = 11
ds = xr.Dataset(
    data_vars={
        "y": ("z", np.random.rand(nz)),
        "lev": ("z", np.arange(nz) * 10),
        # ^ We want this to be a dimension coordinate
    },
)
print(f"ds\n{ds}")
print(f"\nds, 'lev' -> dim coord\n{ds.swap_dims(z='lev')}")

ds2 = (
    ds.swap_dims(z="lev")
    .rename_dims(lev="z")
    .reset_index("lev")
    .reset_coords()
)
print(f"\nds2\n{ds2}")
# ^ This Dataset appears same as the original

print(f"\nds2, 'lev' -> dim coord\n{ds2.swap_dims(z='lev')}")
# ^ Produces a Dataset with dimension coordinate 'lev'
print(f"\nds2 after .swap_dims() applied\n{ds2}")
# ^ `ds2['lev']` now has dimension 'lev' although otherwise same
```


### MVCE confirmation

- [X] Minimal example — the example is as focused as reasonably possible to demonstrate the underlying issue in xarray.
- [X] Complete example — the example is self-contained, including all data and the text of any traceback.
- [X] Verifiable example — the example copy & pastes into an IPython prompt or [Binder notebook](https://mybinder.org/v2/gh/pydata/xarray/main?ur
```

## Librarian submit_result answers (the search guidance)

1. The `swap_dims` method is defined in `/testbed/xarray/core/dataset.py` at lines 3688–3790.

2. The `to_index_variable` method is defined twice in `/testbed/xarray/core/variable.py`:

1. **Line 547–551** — in the `Variable` class (line 276), returns a new `IndexVariable`.
2. **Line 2882–2884** — in the `IndexVariable` class (line 2710), overrides to return `self`.

3. Five test files reference `swap_dims`:

1. **`/testbed/xarray/tests/test_dataset.py`** — Line 2958: `test_swap_dims` method (main Dataset tests, lines 2958–2992).
2. **`/testbed/xarray/tests/test_dataarray.py`** — Line 1771: `test_swap_dims` method (main DataArray tests, lines 1771–1791).
3. **`/testbed/xarray/tests/test_dask.py`** — Line 383 and 1643: references in dask-related test contexts.
4. **`/testbed/xarray/tests/test_sparse.py`** — Line 3192 and 5480: sparse array test references.
5. **`/testbed/xarray/tests/test_units.py`** — Line 383: unit-aware test reference.

## base (PASS) patch

```diff
diff --git a/xarray/core/variable.py b/xarray/core/variable.py
index 5827b90a..d03654e9 100644
--- a/xarray/core/variable.py
+++ b/xarray/core/variable.py
@@ -2881,7 +2881,7 @@ class IndexVariable(Variable):
 
     def to_index_variable(self):
         """Return this variable as an xarray.IndexVariable"""
-        return self
+        return self.copy(deep=False)
 
     to_coord = utils.alias(to_index_variable, "to_coord")
```

## +Librarian (FAIL) patch

```diff
diff --git a/xarray/core/dataset.py b/xarray/core/dataset.py
index c775d05c..564b40a3 100644
--- a/xarray/core/dataset.py
+++ b/xarray/core/dataset.py
@@ -3772,7 +3772,7 @@ class Dataset(
         for k, v in self.variables.items():
             dims = tuple(dims_dict.get(dim, dim) for dim in v.dims)
             if k in result_dims:
-                var = v.to_index_variable()
+                var = v.to_index_variable().copy()
                 var.dims = dims
                 if k in self._indexes:
                     indexes[k] = self._indexes[k]
```
