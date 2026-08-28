# pydata__xarray-3993  (27B / BOAD)

**Failure mode:** stale-negative  
**Note:** lib claimed 'no test uses integrate w/ dim'

- base (no Librarian) `boad_seed0`: passed=True
- +Librarian `boad_librarian_seed0`: passed=False

**Gold/base edited files:** ['xarray/core/dataarray.py', 'xarray/tests/test_dataset.py', 'xarray/tests/test_units.py']

**+Librarian edited files:** ['xarray/core/dataarray.py', 'xarray/tests/test_units.py']

## Issue

```
DataArray.integrate has a 'dim' arg, but Dataset.integrate has a 'coord' arg
This is just a minor gripe but I think it should be fixed.

The API syntax is inconsistent:
```python
ds.differentiate(coord='x')
da.differentiate(coord='x')
ds.integrate(coord='x')
da.integrate(dim='x')   # why dim??
```
It should definitely be `coord` - IMO it doesn't make sense to integrate or differentiate over a dim because a dim by definition has no information about the distance between grid points. I think because the distinction between dims and coords is one of the things that new users have to learn about, we should be strict to not confuse up the meanings in the documentation/API.

The discussion on the original PR [seems to agree](https://github.com/pydata/xarray/pull/2653#discussion_r246164990), so I think this was just an small oversight.

The only question is whether it requires a deprecation cycle?
```

## Librarian submit_result answers (the search guidance)

1. Found `DataArray.integrate` in `/testbed/xarray/core/dataarray.py` lines 3483-3532. The method signature begins at line 3483 and the implementation (which delegates to the Dataset's integrate method) ends at line 3532.

2. Found `Dataset.integrate` in `/testbed/xarray/core/dataset.py` at lines 5966-6023. The helper method `_integrate_one` that it delegates to is at lines 6025-6070.

3. No test files use `integrate` with a `dim` parameter. The `integrate` method signature uses `coord` (not `dim`) as its first parameter. All test usages in `/testbed/xarray/tests/test_dataset.py` (lines 6575-6655) pass the coordinate name as a positional argument, e.g., `da.integrate("x")` or `da.integrate("time", datetime_unit="D")`. The relevant test functions are `test_trapz` (around line 6548) and `test_trapz_datetime` (line 6609).

4. Found two `test_computation` methods in `/testbed/xarray/tests/test_units.py` that use the `integrate` method:

1. **Line 3691** — In class `TestDataArray` (starts at line 2236). The `integrate` method is included as a test parameter at line 3684: `method("integrate", coord="x")`.

2. **Line 5203** — In class `TestDataset` (starts at line 3856). The `integrate` method is included as a test parameter at line 5186: `method("integrate", coord="x")`.

Both tests use `integrate` with `coord="x"` as a keyword argument via the `method()` helper class.

## base (PASS) patch

```diff
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
index 2fef3edb..86a6d7dd 100644
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -3481,21 +3481,27 @@ class DataArray(AbstractArray, DataWithCoords):
         return self._from_temp_dataset(ds)
 
     def integrate(
-        self, dim: Union[Hashable, Sequence[Hashable]], datetime_unit: str = None
+        self,
+        coord: Union[Hashable, Sequence[Hashable]] = None,
+        datetime_unit: str = None,
+        dim: Union[Hashable, Sequence[Hashable]] = None,
     ) -> "DataArray":
         """ integrate the array with the trapezoidal rule.
 
         .. note::
-            This feature is limited to simple cartesian geometry, i.e. dim
+            This feature is limited to simple cartesian geometry, i.e. coord
             must be one dimensional.
 
         Parameters
         ----------
-        dim : hashable, or sequence of hashable
+        coord : hashable, or sequence of hashable
             Coordinate(s) used for the integration.
         datetime_unit : {"Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", \
                          "ps", "fs", "as"}, optional
             Can be used to specify the unit if datetime coordinate is used.
+        dim : hashable, or sequence of hashable
+            Deprecated name for the argument *coord*.
+            Will be removed in a future version.
 
         Returns
         -------
@@ -3528,7 +3534,17 @@ class DataArray(AbstractArray, DataWithCoords):
         array([5.4, 6.6, 7.8])
         Dimensions without coordinates: y
         """
-        ds = self._to_temp_dataset().integrate(dim, datetime_unit)
+        if coord is None and dim is None:
+            raise TypeError("integrate() missing 1 required argument: 'coord'")
+        if coord is not None and dim is not None:
+            raise TypeError("integrate() got multiple values for 'coord'")
+        if dim is not None:
+            warnings.warn(
+                "dim is deprecated, use coord instead.",
+                FutureWarning,
+            )
+            coord = dim
+        ds = self._to_temp_dataset().integrate(coord, datetime_unit)
         return self._from_temp_dataset(ds)
 
     def unify_chunks(self) -> "DataArray":
diff --git a/xarray/tests/test_dataset.py b/xarray/tests/test_dataset.py
index fed90987..f394d791 100644
--- a/xarray/tests/test_dataset.py
+++ b/xarray/tests/test_dataset.py
@@ -6604,6 +6604,28 @@ def test_integrate(dask):
         da.integrate("x2d")
 
 
+def test_integrate_deprecated_dim():
+    da = xr.DataArray(
+        [1, 2, 3],
+        dims=["x"],
+        coords={"x": [0, 1, 3]},
+    )
+
+    # Test that using dim= keyword raises FutureWarning
+    with pytest.warns(FutureWarning, match="dim is deprecated"):
+        da.integrate(dim="x")
+
+    # Test that using coord= keyword does NOT raise warning
+    with warnings.catch_warnings():
+        warnings.simplefilter("error")
+        da.integrate(coord="x")
+
+    # Test that using positional argument does NOT raise warning
+    with warnings.catch_warnings():
+        warnings.simplefilter("error")
+        da.integrate("x")
+
+
 @pytest.mark.parametrize("dask", [True, False])
 @pytest.mark.parametrize("which_datetime", ["np", "cftime"])
 def test_trapz_datetime(dask, which_datetime):
diff --git a/xarray/tests/test_units.py b/xarray/tests/test_units.py
index bb3127e9..76dd830d 100644
--- a/xarray/tests/test_units.py
+++ b/xarray/tests/test_units.py
@
```

## +Librarian (FAIL) patch

```diff
diff --git a/xarray/core/dataarray.py b/xarray/core/dataarray.py
index 2fef3edb..510e6d48 100644
--- a/xarray/core/dataarray.py
+++ b/xarray/core/dataarray.py
@@ -3481,17 +3481,17 @@ class DataArray(AbstractArray, DataWithCoords):
         return self._from_temp_dataset(ds)
 
     def integrate(
-        self, dim: Union[Hashable, Sequence[Hashable]], datetime_unit: str = None
+        self, coord: Union[Hashable, Sequence[Hashable]], datetime_unit: str = None
     ) -> "DataArray":
         """ integrate the array with the trapezoidal rule.
 
         .. note::
-            This feature is limited to simple cartesian geometry, i.e. dim
+            This feature is limited to simple cartesian geometry, i.e. coord
             must be one dimensional.
 
         Parameters
         ----------
-        dim : hashable, or sequence of hashable
+        coord : hashable, or sequence of hashable
             Coordinate(s) used for the integration.
         datetime_unit : {"Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", \
                          "ps", "fs", "as"}, optional
@@ -3528,7 +3528,7 @@ class DataArray(AbstractArray, DataWithCoords):
         array([5.4, 6.6, 7.8])
         Dimensions without coordinates: y
         """
-        ds = self._to_temp_dataset().integrate(dim, datetime_unit)
+        ds = self._to_temp_dataset().integrate(coord, datetime_unit)
         return self._from_temp_dataset(ds)
 
     def unify_chunks(self) -> "DataArray":
diff --git a/xarray/tests/test_units.py b/xarray/tests/test_units.py
index bb3127e9..76dd830d 100644
--- a/xarray/tests/test_units.py
+++ b/xarray/tests/test_units.py
@@ -3681,7 +3681,7 @@ class TestDataArray:
         (
             method("diff", dim="x"),
             method("differentiate", coord="x"),
-            method("integrate", dim="x"),
+            method("integrate", coord="x"),
             method("quantile", q=[0.25, 0.75]),
             method("reduce", func=np.sum, dim="x"),
             pytest.param(lambda x: x.dot(x), id="method_dot"),
```
