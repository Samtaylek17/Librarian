# matplotlib__matplotlib-24026  (35B-A3B / HyperAgent)

**Failure mode:** search-crowdout  
**Note:** 8 lookups, empty final patch

- base (no Librarian) `hyperagent_seed0`: passed=True
- +Librarian `hyperagent_librarian_seed0`: passed=False

**Gold/base edited files:** ['lib/matplotlib/stackplot.py']

**+Librarian edited files:** []

## Issue

```
stackplot should not change Axes cycler
Usecase: I am producing various types of plots (some use rectangle collections, some regular plot-lines, some stacked plots) and wish to keep the colors synchronized across plot types for consistency and ease of comparison.

While `ax.plot()` and `matplotlib.patches.Rectangle()` support supplying a `CN` alias, stackplot throws a ValueError. For example:

```
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy

my_data = numpy.array([[1, 1, 1], [1, 2, 3], [4, 3, 2]])
fig, ax = plt.subplots()
ax.plot([1, 3], [1, 3], color='C0')
ax.add_patch(Rectangle(xy=(1.5, 1.5), width=0.5, height=0.5, facecolor='C1'))
ax.stackplot([1, 2, 3], my_data, colors=['C2', 'C3', 'C4'])
plt.show()
```

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/hmedina/.local/lib/python3.9/site-packages/matplotlib/__init__.py", line 1412, in inner
    return func(ax, *map(sanitize_sequence, args), **kwargs)
  File "/home/hmedina/.local/lib/python3.9/site-packages/matplotlib/stackplot.py", line 73, in stackplot
    axes.set_prop_cycle(color=colors)
  File "/home/hmedina/.local/lib/python3.9/site-packages/matplotlib/axes/_base.py", line 1575, in set_prop_cycle
    prop_cycle = cycler(*args, **kwargs)
  File "/home/hmedina/.local/lib/python3.9/site-packages/matplotlib/rcsetup.py", line 695, in cycler
    vals = validator(vals)
  File "/home/hmedina/.local/lib/python3.9/site-packages/matplotlib/rcsetup.py", line 107, in f
    val = [scalar_validator(v) for v in s
  File 
```

## Librarian submit_result answers (the search guidance)

1. The `stackplot` function is defined in `/testbed/lib/matplotlib/stackplot.py` at lines 16-18 (function signature). The `axes.set_prop_cycle` call is at line 73.

2. Full `stackplot` function body in `/testbed/lib/matplotlib/stackplot.py` lines 16-124. After `axes.set_prop_cycle(color=colors)` at line 73, colors are consumed via `axes._get_lines.get_next_color()` at lines 111 and 120 for each `fill_between` call that creates the stacked area polygons.

3. Found in `/testbed/lib/matplotlib/colors.py`:

- **`_is_nth_color(c)`** (lines 213-215): Detects 'CN' aliases like 'C0', 'C1' using regex `r"\AC[0-9]+\Z"`.
- **`to_rgba(c, alpha=None)`** (lines 265-304): Main conversion function. At line 290-293, it special-cases nth colors by looking up the color from `mpl.rcParams['axes.prop_cycle']` and extracting the actual color string.
- **`to_rgba_array(c, alpha=None)`** (lines 399-480+): Converts arrays of colors, delegates to `to_rgba` for individual color conversion.

4. Found in `/testbed/lib/matplotlib/axes/_base.py`:

- **`_process_plot_var_args.set_prop_cycle`** (lines 234-238): Simple version that wraps a cycler with `itertools.cycle`.
- **`Axes.set_prop_cycle`** (lines 1550-1622): Main method. Accepts either a `Cycler` object or keyword arguments like `color=['red', 'green', 'blue']`. Delegates to the `cycler` package (line 1620: `prop_cycle = cycler(*args, **kwargs)`) and then propagates to `_get_lines` and `_get_patches_for_fill` (lines 1621-1622).

5. Already viewed earlier in this session. Imports in `/testbed/lib/matplotlib/stackplot.py` lines 1-18:
- `import numpy as np` (line 9)
- `from matplotlib import _api` (line 11)
- `__all__ = ['stackplot']` (line 13)
- `def stackplot(axes, x, *args, labels=(), colors=None, baseline='zero', **kwargs):` (lines 16-18)

6. Found `get_next_color` in `/testbed/lib/matplotlib/axes/_base.py` lines 314-318, inside the `_process_plot_var_args` class. It returns the next color from `self.prop_cycler` (an `itertools.cycle` wrapping the prop_cycle Cycler), or defaults to 'k' (black) if 'color' is not in the property keys.

7. Stackplot tests are in `/testbed/lib/matplotlib/tests/test_axes.py`:

- **`test_stackplot`** (lines 2850-2867): Basic stackplot test with labeled data support.
- **`test_stackplot_baseline`** (lines 2870-2892): Tests all four baseline options ('zero', 'sym', 'wiggle', 'weighted_wiggle').

8. Full `stackplot` function in `/testbed/lib/matplotlib/stackplot.py` lines 17-127. Key changes from earlier view: `axes.set_prop_cycle` at line 73 has been replaced with `colors = to_rgba_array(colors)` at line 74, and colors are now consumed via `iter(colors)` (line 113) or a generator using `axes._get_lines.get_next_color()` (line 115).

## base (PASS) patch

```diff
diff --git a/lib/matplotlib/stackplot.py b/lib/matplotlib/stackplot.py
index c580043eeb..ce602f663c 100644
--- a/lib/matplotlib/stackplot.py
+++ b/lib/matplotlib/stackplot.py
@@ -7,8 +7,10 @@ https://stackoverflow.com/q/2225995/
 """
 
 import numpy as np
+from itertools import cycle
 
 from matplotlib import _api
+from matplotlib import colors as mcolors
 
 __all__ = ['stackplot']
 
@@ -70,7 +72,10 @@ def stackplot(axes, x, *args,
 
     labels = iter(labels)
     if colors is not None:
-        axes.set_prop_cycle(color=colors)
+        colors = mcolors.to_rgba_array(colors)
+        color_cycle = cycle(colors)
+    else:
+        color_cycle = None
 
     # Assume data passed has not been 'stacked', so stack it here.
     # We'll need a float buffer for the upcoming calculations.
@@ -108,7 +113,10 @@ def stackplot(axes, x, *args,
         stack += first_line
 
     # Color between x = 0 and the first array.
-    color = axes._get_lines.get_next_color()
+    if color_cycle is not None:
+        color = next(color_cycle)
+    else:
+        color = axes._get_lines.get_next_color()
     coll = axes.fill_between(x, first_line, stack[0, :],
                              facecolor=color, label=next(labels, None),
                              **kwargs)
@@ -117,7 +125,10 @@ def stackplot(axes, x, *args,
 
     # Color between array i-1 and array i
     for i in range(len(y) - 1):
-        color = axes._get_lines.get_next_color()
+        if color_cycle is not None:
+            color = next(color_cycle)
+        else:
+            color = axes._get_lines.get_next_color()
         r.append(axes.fill_between(x, stack[i, :], stack[i + 1, :],
                                    facecolor=color, label=next(labels, None),
                                    **kwargs))
```

## +Librarian (FAIL) patch

```diff
(empty)
```
