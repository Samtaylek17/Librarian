# sphinx-doc__sphinx-9591  (35B-A3B / BOAD)

**Failure mode:** adjacent-module  
**Note:** lib ext/autodoc; gold domains/python.py

- base (no Librarian) `boad_seed0`: passed=True
- +Librarian `boad_librarian_seed0`: passed=False

**Gold/base edited files:** ['sphinx/domains/python.py', 'tests/test_domain_py.py']

**+Librarian edited files:** ['sphinx/ext/autodoc/__init__.py', 'tests/roots/test-ext-autodoc/target/properties.py', 'tests/test_ext_autodoc.py', 'tests/test_ext_autodoc_autoclass.py', 'tests/test_ext_autodoc_autoproperty.py', 'tests/test_ext_autodoc_configs.py']

## Issue

```
Cross-references don't work in property's type annotations
### Describe the bug

A documented type in property's type annotation does not get cross-referenced:
```py
from typing import Optional


class Point:
    """
    A class representing a point.

    Attributes:
        x: Position X.
        y: Position Y.
    """
    x: int
    y: int


class Square:
    """A class representing a square figure."""
    #: Square's start position (top-left corner).
    start: Point
    #: Square width.
    width: int
    #: Square height.
    height: int

    @property
    def end(self) -> Point:
        """Square's end position (bottom-right corner)."""
        return Point(self.start.x + self.width, self.start.y + self.height)


class Rectangle:
    """
    A class representing a square figure.

    Attributes:
        start: Rectangle's start position (top-left corner).
        width: Rectangle width.
        height: Rectangle width.
    """
    start: Point
    width: int
    height: int

    @property
    def end(self) -> Point:
        """Rectangle's end position (bottom-right corner)."""
        return Point(self.start.x + self.width, self.start.y + self.height)
```

### How to Reproduce

```
$ git clone https://github.com/jack1142/sphinx-issue-9585
$ cd sphinx-issue-9585
$ pip install sphinx
$ cd docs
$ make html
$ # open _build/html/index.html and see the issue
```


### Expected behavior

I expected the documented type in property's type annotation to be cross-referenced.

### Your project

https://github.
```

## Librarian submit_result answers (the search guidance)

1. The `PropertyDocumenter` class is defined in `/testbed/sphinx/ext/autodoc/__init__.py` at lines 2664–2739.

2. Both functions are defined in `/testbed/sphinx/util/typing.py`:

- `restify` at lines 107–140 — converts a Python class to a reST reference.
- `stringify` at lines 294–332 — stringifies a type annotation object. In `sphinx/ext/autodoc/__init__.py` (line 37), it is imported as `stringify_typehint` via `from sphinx.util.typing import stringify as stringify_typehint`.

## base (PASS) patch

```diff
diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py
index e8330e81c..9875a9f4b 100644
--- a/sphinx/domains/python.py
+++ b/sphinx/domains/python.py
@@ -861,7 +861,8 @@ class PyProperty(PyObject):
 
         typ = self.options.get('type')
         if typ:
-            signode += addnodes.desc_annotation(typ, ': ' + typ)
+            annotations = _parse_annotation(typ, self.env)
+            signode += addnodes.desc_annotation(typ, '', nodes.Text(': '), *annotations)
 
         return fullname, prefix
diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py
index 8b72f8b7a..067febcf4 100644
--- a/tests/test_domain_py.py
+++ b/tests/test_domain_py.py
@@ -833,13 +833,15 @@ def test_pyproperty(app):
                 entries=[('single', 'prop1 (Class property)', 'Class.prop1', '', None)])
     assert_node(doctree[1][1][1], ([desc_signature, ([desc_annotation, "abstract property "],
                                                      [desc_name, "prop1"],
-                                                     [desc_annotation, ": str"])],
+                                                     [desc_annotation, (": ",
+                                                                        [pending_xref, "str"])])],
                                    [desc_content, ()]))
     assert_node(doctree[1][1][2], addnodes.index,
                 entries=[('single', 'prop2 (Class property)', 'Class.prop2', '', None)])
     assert_node(doctree[1][1][3], ([desc_signature, ([desc_annotation, "class property "],
                                                      [desc_name, "prop2"],
-                                                     [desc_annotation, ": str"])],
+                                                     [desc_annotation, (": ",
+                                                                        [pending_xref, "str"])])],
                                    [desc_content, ()]))
     assert 'Class.prop1' in domain.objects
     assert domain.objects['Class.prop1'] == ('index', 'Class.prop1', 'property', False)
```

## +Librarian (FAIL) patch

```diff
diff --git a/sphinx/ext/autodoc/__init__.py b/sphinx/ext/autodoc/__init__.py
index 938f976cd..a6cd71dcd 100644
--- a/sphinx/ext/autodoc/__init__.py
+++ b/sphinx/ext/autodoc/__init__.py
@@ -2728,7 +2728,7 @@ class PropertyDocumenter(DocstringStripSignatureMixin, ClassLevelDocumenter):  #
                 signature = inspect.signature(func,
                                               type_aliases=self.config.autodoc_type_aliases)
                 if signature.return_annotation is not Parameter.empty:
-                    objrepr = stringify_typehint(signature.return_annotation)
+                    objrepr = restify(signature.return_annotation)
                     self.add_line('   :type: ' + objrepr, sourcename)
             except TypeError as exc:
                 logger.warning(__("Failed to get a function signature for %s: %s"),
diff --git a/tests/roots/test-ext-autodoc/target/properties.py b/tests/roots/test-ext-autodoc/target/properties.py
index 561daefb8..2bd35613a 100644
--- a/tests/roots/test-ext-autodoc/target/properties.py
+++ b/tests/roots/test-ext-autodoc/target/properties.py
@@ -1,3 +1,8 @@
+class Point:
+    """A class representing a point."""
+    pass
+
+
 class Foo:
     """docstring"""
 
@@ -9,3 +14,7 @@ class Foo:
     @property
     def prop2(self) -> int:
         """docstring"""
+
+    @property
+    def prop3(self) -> Point:
+        """docstring"""
diff --git a/tests/test_ext_autodoc.py b/tests/test_ext_autodoc.py
index 299c1c681..a86ff0c87 100644
--- a/tests/test_ext_autodoc.py
+++ b/tests/test_ext_autodoc.py
@@ -1084,7 +1084,7 @@ def test_autodoc_cached_property(app):
         '',
         '   .. py:property:: Foo.prop',
         '      :module: target.cached_property',
-        '      :type: int',
+        '      :type: :class:`int`',
         '',
     ]
diff --git a/tests/test_ext_autodoc_autoclass.py b/tests/test_ext_autodoc_autoclass.py
index 24617bf0a..929e04097 100644
--- a/tests/test_ext_autodoc_autoclass.py
+++ b/tests/test_ext_autodoc_autoclass.py
@@ -214,7 +214,7 @@ def test_properties(app):
         '',
         '   .. py:property:: Foo.prop1',
         '      :module: target.properties',
-        '      :type: int',
+        '      :type: :class:`int`',
         '',
         '      docstring',
         '',
@@ -222,7 +222,14 @@ def test_properties(app):
         '   .. py:property:: Foo.prop2',
         '      :module: target.properties',
         '      :classmethod:',
-        '      :type: int',
+        '      :type: :class:`int`',
+        '',
+        '      docstring',
+        '',
+        '',
+        '   .. py:property:: Foo.prop3',
+        '      :module: target.properties',
+        '      :type: :class:`target.properties.Point`',
         '',
         '      docstring',
         '',
diff --git a/tests/test_ext_autodoc_autoproperty.py b/tests/test_ext_autodoc_autoproperty.py
index 47528a99d..c766cb99c 100644
--- a/tests/test_ext_autodoc_autoproperty.py
+++ b/tests/test_ext_autodoc_autoproperty.py
@@ -23,7 +23,7 @@ def test_properties(app):
         '',
         '.. py:property:: Foo.prop1',
         '   :module: target.properties',
-        '   :type: int',
+        '   :type: :class:`int`',
         '',
         '   docstring',
         '',
@@ -38,7 +38,22 @@ def test_class_properties(app):
         '.. py:property:: Foo.prop2',
         '   :module: target.properties',
         '   :classmethod:',
-        '   :type: int',
+        '   :type: :class:`int`',
+        '',
+        ' 
```
