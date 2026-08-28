# django__django-12193  (27B / HyperAgent)

**Failure mode:** adjacent-module  
**Note:** lib postgres/forms/array.py; gold forms/widgets.py

- base (no Librarian) `hyperagent_seed0`: passed=True
- +Librarian `hyperagent_librarian_seed0`: passed=False

**Gold/base edited files:** ['django/forms/widgets.py']

**+Librarian edited files:** ['django/contrib/postgres/forms/array.py']

## Issue

```
SplitArrayField with BooleanField always has widgets checked after the first True value.
Description
	 
		(last modified by Peter Andersen)
	 
When providing a SplitArrayField BooleanField with preexisting data, the final_attrs dict is updated to include 'checked': True after the for loop has reached the first True value in the initial data array. Once this occurs every widget initialized after that defaults to checked even though the backing data may be False. This is caused by the CheckboxInput widget's get_context() modifying the attrs dict passed into it. This is the only widget that modifies the attrs dict passed into its get_context().
CheckboxInput setting attrs['checked'] to True: ​https://github.com/django/django/blob/master/django/forms/widgets.py#L527
```

## Librarian submit_result answers (the search guidance)

1. `SplitArrayField` is defined in `/testbed/django/contrib/postgres/forms/array.py` at lines 168–234. It's a Django form field class that extends `forms.Field` and handles array input with a `SplitArrayWidget`.

2. `SplitArrayWidget` is defined in `/testbed/django/contrib/postgres/forms/array.py` at lines 105–165. It's a Django widget class extending `forms.Widget` that renders multiple sub-widgets for array input.

3. `CheckboxInput.get_context` is defined in `/testbed/django/forms/widgets.py` at lines 523–528. It's a method of the `CheckboxInput` class that adds a `checked` attribute when `check_test(value)` returns True, then delegates to the parent's `get_context`.

4. `SplitArrayWidget.get_context` is at `/testbed/django/contrib/postgres/forms/array.py` lines 133–152. Already shown above from earlier in this session.

5. Tests for `SplitArrayField` and `SplitArrayWidget` are in `/testbed/tests/postgres_tests/test_array.py`:

- **`TestSplitFormField`** (lines 930–1067): Tests for `SplitArrayField` including validation, required fields, `remove_trailing_nulls`, rendering, `has_changed`, and integration with ModelForm.
- **`TestSplitFormWidget`** (lines 1069–1132): Tests for `SplitArrayWidget` including `get_context`, rendering, attrs handling, and `value_omitted_from_data`.

6. `CheckboxInput.check_test` is an instance attribute set in `__init__` at `/testbed/django/forms/widgets.py` line 515. It defaults to the `boolean_check` function (lines 503–504) unless a custom callable is passed. It's used in `get_context` (line 524) to determine whether to add the `checked` attribute.

## base (PASS) patch

```diff
diff --git a/django/forms/widgets.py b/django/forms/widgets.py
index 6fe220bea7..46f494e702 100644
--- a/django/forms/widgets.py
+++ b/django/forms/widgets.py
@@ -524,7 +524,7 @@ class CheckboxInput(Input):
         if self.check_test(value):
             if attrs is None:
                 attrs = {}
-            attrs['checked'] = True
+            attrs = {**attrs, 'checked': True}
         return super().get_context(name, value, attrs)
 
     def value_from_datadict(self, data, files, name):
```

## +Librarian (FAIL) patch

```diff
diff --git a/django/contrib/postgres/forms/array.py b/django/contrib/postgres/forms/array.py
index 2e19cd574a..341f563e3b 100644
--- a/django/contrib/postgres/forms/array.py
+++ b/django/contrib/postgres/forms/array.py
@@ -145,9 +145,11 @@ class SplitArrayWidget(forms.Widget):
             except IndexError:
                 widget_value = None
             if id_:
-                final_attrs = {**final_attrs, 'id': '%s_%s' % (id_, i)}
+                widget_attrs = {**final_attrs, 'id': '%s_%s' % (id_, i)}
+            else:
+                widget_attrs = final_attrs.copy()
             context['widget']['subwidgets'].append(
-                self.widget.get_context(name + '_%s' % i, widget_value, final_attrs)['widget']
+                self.widget.get_context(name + '_%s' % i, widget_value, widget_attrs)['widget']
             )
         return context
```
