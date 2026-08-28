# sympy__sympy-21379  (35B-A3B / HyperAgent)

**Failure mode:** symptom-vs-root  
**Note:** lib -> hyperbolic.py (symptom); gold core/mod.py (root)

- base (no Librarian) `hyperagent_seed0`: passed=True
- +Librarian `hyperagent_librarian_seed0`: passed=False

**Gold/base edited files:** ['sympy/core/mod.py']

**+Librarian edited files:** ['sympy/functions/elementary/hyperbolic.py', 'sympy/functions/elementary/piecewise.py']

## Issue

```
Unexpected `PolynomialError` when using simple `subs()` for particular expressions
I am seeing weird behavior with `subs` for particular expressions with hyperbolic sinusoids with piecewise arguments. When applying `subs`, I obtain an unexpected `PolynomialError`. For context, I was umbrella-applying a casting from int to float of all int atoms for a bunch of random expressions before using a tensorflow lambdify to avoid potential tensorflow type errors. You can pretend the expression below has a `+ 1` at the end, but below is the MWE that I could produce.

See the expression below, and the conditions in which the exception arises.

Sympy version: 1.8.dev

```python
from sympy import *
from sympy.core.cache import clear_cache

x, y, z = symbols('x y z')

clear_cache()
expr = exp(sinh(Piecewise((x, y > x), (y, True)) / z))
# This works fine
expr.subs({1: 1.0})

clear_cache()
x, y, z = symbols('x y z', real=True)
expr = exp(sinh(Piecewise((x, y > x), (y, True)) / z))
# This fails with "PolynomialError: Piecewise generators do not make sense"
expr.subs({1: 1.0})  # error
# Now run it again (isympy...) w/o clearing cache and everything works as expected without error
expr.subs({1: 1.0})
```

I am not really sure where the issue is, but I think it has something to do with the order of assumptions in this specific type of expression. Here is what I found-

- The error only (AFAIK) happens with `cosh` or `tanh` in place of `sinh`, otherwise it succeeds
- The error goes away if removing the division by `z`
- The error goes away if removing `exp` (but s
```

## Librarian submit_result answers (the search guidance)

1. Found `_eval_is_real` in `/testbed/sympy/functions/elementary/hyperbolic.py` lines 243-251, inside the `sinh` class (line 72). The method checks if the argument is real, and if not, extracts the imaginary part and checks if `(im % pi).is_zero`.

2. Found `Mod.eval` in `/testbed/sympy/core/mod.py` lines 38-221. Line 169 specifically contains `G = gcd(p, q)` which extracts the GCD for simplification. The method handles various cases for computing modulo: numeric evaluation, denesting, separating Add/Mul terms, and GCD extraction.

3. Found the `PolynomialError("Piecewise generators do not make sense")` raised in `/testbed/sympy/polys/polytools.py` at line 4399, inside the `_parallel_poly_from_expr` function (line 4343). The error is triggered when a generator in `opt.gens` is an instance of `Piecewise`.

4. Three `_eval_is_real` methods found in `/testbed/sympy/functions/elementary/hyperbolic.py`:

1. **sinh** (line 243-251): Returns True if arg is real; otherwise checks if `(im % pi).is_zero`
2. **cosh** (line 429-440): Returns True if arg is real OR imaginary; otherwise checks if `(im % pi).is_zero`
3. **tanh** (line 653-666): Returns True if arg is real; handles special case where `re == 0 and im % pi == pi/2` (returns None); otherwise checks if `(im % (pi/2)).is_zero`

5. Complete class definitions found in `/testbed/sympy/functions/elementary/hyperbolic.py`:

1. **sinh** (lines 72-274): Includes `eval`, `as_real_imag`, `_eval_is_real` (lines 243-251), and other property methods.

2. **cosh** (lines 275-503): Includes `eval`, `as_real_imag`, `_eval_is_real` (lines 429-440), `_eval_is_positive`, `_eval_is_nonnegative`, and other property methods.

3. **tanh** (lines 505-698): Includes `eval`, `as_real_imag`, `_eval_is_real` (lines 653-666), `_eval_is_finite`, and other property methods.

6. All three `_eval_is_real` methods from `/testbed/sympy/functions/elementary/hyperbolic.py` (no changes in these line ranges per freshness report):

1. **sinh._eval_is_real** (lines 243-251): Checks `arg.is_real`, then uses `(im % pi).is_zero`
2. **cosh._eval_is_real** (lines 429-440): Checks `arg.is_real or arg.is_imaginary`, then uses `(im % pi).is_zero`
3. **tanh._eval_is_real** (lines 653-666): Checks `arg.is_real`, handles special case `re == 0 and im % pi == pi/2` returning None, then uses `(im % (pi/2)).is_zero`

7. Lines 243-260 in `/testbed/sympy/functions/elementary/hyperbolic.py` show the `_eval_is_real` method for the `sinh` class, which includes a try/except block catching `PolynomialError` when checking if `(im % pi).is_zero`.

8. Lines 429-445 in `/testbed/sympy/functions/elementary/hyperbolic.py` show the `_eval_is_real` method for the `cosh` class, which includes a try/except block catching `PolynomialError` when checking if `(im % pi).is_zero`.

9. Lines 653-680 in `/testbed/sympy/functions/elementary/hyperbolic.py` show the `_eval_is_real` method for the `tanh` class (lines 666-685), which includes try/except blocks catching `PolynomialError` when checking the imaginary part conditions.

10. Lines 3058-3066 in `/testbed/sympy/polys/tests/test_polytools.py` are part of the `test_cancel` function (line 2968), testing issue 7022 which involves Piecewise expressions with cancel().

11. Imports at the top of `/testbed/sympy/functions/elementary/hyperbolic.py` (lines 1-12) include `sympy.core`, `sympy.core.function`, `sympy.functions.combinatorial.factorials`, `sympy.functions.elementary.exponential`, `sympy.functions.elementary.miscellaneous`, `sympy.functions.elementary.integers`, `sympy.core.logic`, and notably `sympy.polys.polyerrors.PolynomialError` (line 12).

12. Git diff for `/testbed/sympy/functions/elementary/hyperbolic.py` shows:

1. Added import: `from sympy.polys.polyerrors import PolynomialError` (line 12)

2. **sinh._eval_is_real** (lines 249-255): Wrapped `(im%pi).is_zero` in try/except for PolynomialError

3. **cosh._eval_is_real** (lines 441-447): Wrapped `(im%pi).is_zero` in try/except for PolynomialError

4. **cosh._eval_is_positive** (lines 455-458): Wrapped `y % (2*pi)` in try/except, returns None on error

5. **cosh._eval_is_nonnegative** (lines 485-488): Wrapped `y % (2*pi)` in try/except, returns None on error

6. **tanh._eval_is_real** (lines 671-685): Wrapped `im % pi == pi/2` and `(im % (pi/2)).is_zero` in try/except blocks

## base (PASS) patch

```diff
diff --git a/sympy/core/mod.py b/sympy/core/mod.py
index b4f3de2523..bf503d715f 100644
--- a/sympy/core/mod.py
+++ b/sympy/core/mod.py
@@ -40,6 +40,7 @@ def eval(cls, p, q):
         from sympy.core.mul import Mul
         from sympy.core.singleton import S
         from sympy.core.exprtools import gcd_terms
+        from sympy.polys.polyerrors import PolynomialError
         from sympy.polys.polytools import gcd
 
         def doit(p, q):
@@ -166,7 +167,10 @@ def doit(p, q):
         # XXX other possibilities?
 
         # extract gcd; any further simplification should be done by the user
-        G = gcd(p, q)
+        try:
+            G = gcd(p, q)
+        except PolynomialError:
+            G = S.One
         if G != 1:
             p, q = [
                 gcd_terms(i/G, clear=False, fraction=False) for i in (p, q)]
```

## +Librarian (FAIL) patch

```diff
diff --git a/sympy/functions/elementary/hyperbolic.py b/sympy/functions/elementary/hyperbolic.py
index 860989f7c1..465effeb59 100644
--- a/sympy/functions/elementary/hyperbolic.py
+++ b/sympy/functions/elementary/hyperbolic.py
@@ -9,6 +9,7 @@
 from sympy.functions.elementary.integers import floor
 
 from sympy.core.logic import fuzzy_or, fuzzy_and
+from sympy.polys.polyerrors import PolynomialError
 
 
 
@@ -248,7 +249,10 @@ def _eval_is_real(self):
         # if `im` is of the form n*pi
         # else, check if it is a number
         re, im = arg.as_real_imag()
-        return (im%pi).is_zero
+        try:
+            return (im%pi).is_zero
+        except PolynomialError:
+            return None
 
     def _eval_is_extended_real(self):
         if self.args[0].is_extended_real:
@@ -437,7 +441,10 @@ def _eval_is_real(self):
         # the imaginary part can be an expression like n*pi
         # if not, check if the imaginary part is a number
         re, im = arg.as_real_imag()
-        return (im%pi).is_zero
+        try:
+            return (im%pi).is_zero
+        except PolynomialError:
+            return None
 
     def _eval_is_positive(self):
         # cosh(x+I*y) = cos(y)*cosh(x) + I*sin(y)*sinh(x)
@@ -448,7 +455,10 @@ def _eval_is_positive(self):
         z = self.args[0]
 
         x, y = z.as_real_imag()
-        ymod = y % (2*pi)
+        try:
+            ymod = y % (2*pi)
+        except PolynomialError:
+            return None
 
         yzero = ymod.is_zero
         # shortcut if ymod is zero
@@ -475,7 +485,10 @@ def _eval_is_nonnegative(self):
         z = self.args[0]
 
         x, y = z.as_real_imag()
-        ymod = y % (2*pi)
+        try:
+            ymod = y % (2*pi)
+        except PolynomialError:
+            return None
 
         yzero = ymod.is_zero
         # shortcut if ymod is zero
@@ -658,12 +671,18 @@ def _eval_is_real(self):
         re, im = arg.as_real_imag()
 
         # if denom = 0, tanh(arg) = zoo
-        if re == 0 and im % pi == pi/2:
+        try:
+            if re == 0 and im % pi == pi/2:
+                return None
+        except PolynomialError:
             return None
 
         # check if im is of the form n*pi/2 to make sin(2*im) = 0
         # if not, im could be a number, return False in that case
-        return (im % (pi/2)).is_zero
+        try:
+            return (im % (pi/2)).is_zero
+        except PolynomialError:
+            return None
 
     def _eval_is_extended_real(self):
         if self.args[0].is_extended_real:
diff --git a/sympy/functions/elementary/piecewise.py b/sympy/functions/elementary/piecewise.py
index 91bfceafaa..d0f161c369 100644
--- a/sympy/functions/elementary/piecewise.py
+++ b/sympy/functions/elementary/piecewise.py
@@ -57,6 +57,9 @@ def __iter__(self):
     def _eval_simplify(self, **kwargs):
         return self.func(*[a.simplify(**kwargs) for a in self.args])
 
+    def _eval_is_meromorphic(self, x, a):
+        return self.expr._eval_is_meromorphic(x, a)
+
 class Piecewise(Function):
     """
     Represents a piecewise function.
```
