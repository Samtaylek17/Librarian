# pylint-dev__pylint-7080  (35B-A3B / BOAD)

**Failure mode:** adjacent-module  
**Note:** lib pylinter.py; gold expand_modules.py

- base (no Librarian) `boad_seed0`: passed=True
- +Librarian `boad_librarian_seed0`: passed=False

**Gold/base edited files:** ['pylint/lint/expand_modules.py', 'tests/lint/unittest_expand_modules.py']

**+Librarian edited files:** ['pylint/lint/pylinter.py']

## Issue

```
`--recursive=y` ignores `ignore-paths`
### Bug description

When running recursively, it seems `ignore-paths` in my settings in pyproject.toml is completely ignored

### Configuration

```ini
[tool.pylint.MASTER]
ignore-paths = [
  # Auto generated
  "^src/gen/.*$",
]
```


### Command used

```shell
pylint --recursive=y src/
```


### Pylint output

```shell
************* Module region_selection
src\region_selection.py:170:0: R0914: Too many local variables (17/15) (too-many-locals)
************* Module about
src\gen\about.py:2:0: R2044: Line with empty comment (empty-comment)
src\gen\about.py:4:0: R2044: Line with empty comment (empty-comment)
src\gen\about.py:57:0: C0301: Line too long (504/120) (line-too-long)
src\gen\about.py:12:0: C0103: Class name "Ui_AboutAutoSplitWidget" doesn't conform to '_?_?[a-zA-Z]+?$' pattern (invalid-name)
src\gen\about.py:12:0: R0205: Class 'Ui_AboutAutoSplitWidget' inherits from object, can be safely removed from bases in python3 (useless-object-inheritance)
src\gen\about.py:13:4: C0103: Method name "setupUi" doesn't conform to snake_case naming style (invalid-name)
src\gen\about.py:13:22: C0103: Argument name "AboutAutoSplitWidget" doesn't conform to snake_case naming style (invalid-name)
src\gen\about.py:53:4: C0103: Method name "retranslateUi" doesn't conform to snake_case naming style (invalid-name)
src\gen\about.py:53:28: C0103: Argument name "AboutAutoSplitWidget" doesn't conform to snake_case naming style (invalid-name)
src\gen\about.py:24:8: W0201: Attribute 'ok_button' defined outside __init__ 
```

## Librarian submit_result answers (the search guidance)

1. - `_discover_files` is defined in `/testbed/pylint/lint/pylinter.py` at lines 585-624 (method of the Pylinter class).
- `_is_ignored_file` is **not** defined in `pylint/lint/pylinter.py` — it is imported from `pylint.lint.expand_modules` (line 37 of pylinter.py). The actual definition is in `/testbed/pylint/lint/expand_modules.py` at lines 49-60.

## base (PASS) patch

```diff
diff --git a/pylint/lint/expand_modules.py b/pylint/lint/expand_modules.py
index 5cacc037..9519e985 100644
--- a/pylint/lint/expand_modules.py
+++ b/pylint/lint/expand_modules.py
@@ -41,9 +41,46 @@ def get_python_path(filepath: str) -> str:
             return os.getcwd()
 
 
-def _is_in_ignore_list_re(element: str, ignore_list_re: list[Pattern[str]]) -> bool:
-    """Determines if the element is matched in a regex ignore-list."""
-    return any(file_pattern.match(element) for file_pattern in ignore_list_re)
+def _is_in_ignore_list_re(
+    element: str, ignore_list_re: list[Pattern[str]], match_dir: bool = False
+) -> bool:
+    """Determines if the element is matched in a regex ignore-list.
+
+    :param element: The element to check
+    :param ignore_list_re: The list of regex patterns to check against
+    :param match_dir: If True, also check if the element matches when a trailing
+        separator is added. This is useful for matching directories against patterns
+        like "^src/gen/.*$" which should also match "src/gen" (the directory itself).
+    """
+    if any(file_pattern.match(element) for file_pattern in ignore_list_re):
+        return True
+    if match_dir:
+        # Also check with a trailing separator to match directories
+        # against patterns like "^src/gen/.*$" which should match "src/gen"
+        element_with_sep = os.path.join(element, "")
+        if any(file_pattern.match(element_with_sep) for file_pattern in ignore_list_re):
+            return True
+    # Also try matching against the path relative to the current working directory.
+    # This allows patterns like "^src/gen/.*$" to match absolute paths like
+    # "/home/user/project/src/gen/about.py" when the cwd is "/home/user/project".
+    try:
+        relative_element = os.path.relpath(element)
+        if relative_element != element:
+            if any(
+                file_pattern.match(relative_element) for file_pattern in ignore_list_re
+            ):
+                return True
+            if match_dir:
+                relative_element_with_sep = os.path.join(relative_element, "")
+                if any(
+                    file_pattern.match(relative_element_with_sep)
+                    for file_pattern in ignore_list_re
+                ):
+                    return True
+    except ValueError:
+        # relpath can raise ValueError on Windows if the path is on a different drive
+        pass
+    return False
 
 
 def _is_ignored_file(
@@ -56,7 +93,7 @@ def _is_ignored_file(
     return (
         basename in ignore_list
         or _is_in_ignore_list_re(basename, ignore_list_re)
-        or _is_in_ignore_list_re(element, ignore_list_paths_re)
+        or _is_in_ignore_list_re(element, ignore_list_paths_re, match_dir=True)
     )
diff --git a/tests/lint/unittest_expand_modules.py b/tests/lint/unittest_expand_modules.py
index 15f72d0c..5ffc25db 100644
--- a/tests/lint/unittest_expand_modules.py
+++ b/tests/lint/unittest_expand_modules.py
@@ -134,11 +134,11 @@ class TestExpandModules(CheckerTestCase):
         "files_or_modules,expected",
         [
             ([__file__], []),
+            # When ignore_paths matches files inside a directory, the directory itself
+            # is also ignored (so no modules from that directory are returned)
             (
                 [str(Path(__file__).parent)],
-                [
-                    init_of_package,
-                ],
+                [],
             ),
    
```

## +Librarian (FAIL) patch

```diff
diff --git a/pylint/lint/pylinter.py b/pylint/lint/pylinter.py
index e2bbf061..1533f932 100644
--- a/pylint/lint/pylinter.py
+++ b/pylint/lint/pylinter.py
@@ -602,6 +602,11 @@ class PyLinter(
                         self.config.ignore,
                         self.config.ignore_patterns,
                         self.config.ignore_paths,
+                    ) or _is_ignored_file(
+                        root + os.sep,
+                        self.config.ignore,
+                        self.config.ignore_patterns,
+                        self.config.ignore_paths,
                     ):
                         skip_subtrees.append(root)
                         continue
```
