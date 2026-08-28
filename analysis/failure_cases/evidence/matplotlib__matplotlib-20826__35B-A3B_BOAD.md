# matplotlib__matplotlib-20826  (35B-A3B / BOAD)

**Failure mode:** adjacent-module  
**Note:** lib axes/_base.py; gold axis.py

- base (no Librarian) `boad_seed0`: passed=True
- +Librarian `boad_librarian_seed0`: passed=False

**Gold/base edited files:** ['lib/matplotlib/axis.py']

**+Librarian edited files:** ['lib/matplotlib/axes/_base.py']

## Issue

```
ax.clear() adds extra ticks, un-hides shared-axis tick labels
### Bug report

**Bug summary**

When using shared axes (e.g. from `plt.subplots(2, 2, sharex=True, sharey=True)`), calling `ax.clear()` causes ticks and tick labels to be shown that should be hidden. The axes are still linked, though (e.g. adjusting the plotting range on one subplot adjusts the others as well). This is a behavior change between matplotlib 3.4.1 and 3.4.2.

**Code for reproduction**

This code produces different results with matplotlib 3.4.1 and 3.4.2:

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

x = np.arange(0.0, 2*np.pi, 0.01)
y = np.sin(x)

for ax in axes.flatten():
    ax.clear()
    ax.plot(x, y)
```

This example is of course silly, but I use the general pattern when making animations with FuncAnimation, where my plotting function is a complex module which doesn't facilitate blitting, so I clear and re-use the axes for each frame of the animation.

**Actual outcome**

This is the plot produced with matplotlib 3.4.2:

![matplotlib-3 4 2](https://user-images.githubusercontent.com/23462789/126717195-a974fcf6-52d6-465b-841e-4f8172964dcd.png)

The presence of tick labels that should be hidden by virtue of the shared axes is the clearest problem in this plot, but there are also ticks that appear along the top and right side of each subplot which are not present in the example below (and not part of the default plotting style, IIRC).

The top and right-side ticks also appear when not usin
```

## Librarian submit_result answers (the search guidance)

1. The `clear()` method in matplotlib axes is defined in `/testbed/lib/matplotlib/axes/_base.py` at lines 1481-1483. It is a thin wrapper that delegates to `self.cla()`.

2. The `cla()` method in matplotlib axes is defined in `/testbed/lib/matplotlib/axes/_base.py` at lines 1182-1320. It performs a comprehensive reset of the axes including clearing axes, spines, callbacks, limits, titles, and patch.

3. The `sharex()` method in matplotlib axes is defined in `/testbed/lib/matplotlib/axes/_base.py` at lines 1144-1161. It shares the x-axis with another axes object by joining shared axes, syncing tickers, and copying limits and scale.

4. The `clear()` method for XAxis/YAxis (via the base Axis class) is defined in `/testbed/lib/matplotlib/axis.py` at lines 788-816. It resets axis properties including label, scale, locators, formatters, ticks, grid, units, and callbacks.

5. The `reset_ticks()` method is defined in `/testbed/lib/matplotlib/axis.py` at lines 823-841. It re-initializes the major and minor Tick lists by deleting cached tick attributes and resetting clip paths.

6. The `Ticker` class is defined in `/testbed/lib/matplotlib/axis.py` at lines 527-566. It is a container for objects defining tick position (locator) and format (formatter).

7. The `xaxis` is not a property but an instance attribute in matplotlib axes. It is created in the `_init_axis()` method at line 743 in `/testbed/lib/matplotlib/axes/_base.py`, where it is assigned as `self.xaxis = maxis.XAxis(self)`.

8. The `major` attribute is defined in the `Axis.__init__()` method at line 654 in `/testbed/lib/matplotlib/axis.py`. It is set as `self.major = Ticker()`. Since `XAxis` inherits from `Axis`, this is where the attribute originates.

## base (PASS) patch

```diff
diff --git a/lib/matplotlib/axis.py b/lib/matplotlib/axis.py
index 887cf97ef1..bbfcc0d1c6 100644
--- a/lib/matplotlib/axis.py
+++ b/lib/matplotlib/axis.py
@@ -774,13 +774,11 @@ class Axis(martist.Artist):
                 *self.get_major_ticks(), *self.get_minor_ticks()]
 
     def _reset_major_tick_kw(self):
-        self._major_tick_kw.clear()
         self._major_tick_kw['gridOn'] = (
                 mpl.rcParams['axes.grid'] and
                 mpl.rcParams['axes.grid.which'] in ('both', 'major'))
 
     def _reset_minor_tick_kw(self):
-        self._minor_tick_kw.clear()
         self._minor_tick_kw['gridOn'] = (
                 mpl.rcParams['axes.grid'] and
                 mpl.rcParams['axes.grid.which'] in ('both', 'minor'))
```

## +Librarian (FAIL) patch

```diff
diff --git a/lib/matplotlib/axes/_base.py b/lib/matplotlib/axes/_base.py
index 802fd3c997..4d33dc9010 100644
--- a/lib/matplotlib/axes/_base.py
+++ b/lib/matplotlib/axes/_base.py
@@ -1192,12 +1192,20 @@ class _AxesBase(martist.Artist):
         xaxis_visible = self.xaxis.get_visible()
         yaxis_visible = self.yaxis.get_visible()
 
+        # stash the current tick params
+        xaxis_tick_kw = self.xaxis._major_tick_kw.copy()
+        yaxis_tick_kw = self.yaxis._major_tick_kw.copy()
+
         self.xaxis.clear()
         self.yaxis.clear()
 
         for name, spine in self.spines.items():
             spine.clear()
 
+        # restore the tick params (after spine.clear() which may call axis.clear())
+        self.xaxis._major_tick_kw.update(xaxis_tick_kw)
+        self.yaxis._major_tick_kw.update(yaxis_tick_kw)
+
         self.ignore_existing_data_limits = True
         self.callbacks = cbook.CallbackRegistry()
```
