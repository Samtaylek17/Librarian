# django__django-13158  (35B-A3B / HyperAgent)

**Failure mode:** adjacent-module  
**Note:** lib sql/query.py; gold sql/compiler.py

- base (no Librarian) `hyperagent_seed0`: passed=True
- +Librarian `hyperagent_librarian_seed0`: passed=False

**Gold/base edited files:** ['django/db/models/sql/compiler.py']

**+Librarian edited files:** ['django/db/models/sql/query.py']

## Issue

```
QuerySet.none() on combined queries returns all results.
Description
	
I came across this issue on Stack Overflow. I'm not 100% sure it's a bug, but it does seem strange. With this code (excuse the bizarre example filtering):
class Publication(models.Model):
	pass
class Article(models.Model):
	publications = models.ManyToManyField(to=Publication, blank=True, null=True)
class ArticleForm(forms.ModelForm):
	publications = forms.ModelMultipleChoiceField(
		Publication.objects.filter(id__lt=2) | Publication.objects.filter(id__gt=5),
		required=False,
	)
	class Meta:
		model = Article
		fields = ["publications"]
class ArticleAdmin(admin.ModelAdmin):
	form = ArticleForm
This works well. However, changing the ModelMultipleChoiceField queryset to use union() breaks things.
publications = forms.ModelMultipleChoiceField(
	Publication.objects.filter(id__lt=2).union(
		Publication.objects.filter(id__gt=5)
	),
	required=False,
)
The form correctly shows only the matching objects. However, if you submit this form while empty (i.e. you didn't select any publications), ALL objects matching the queryset will be added. Using the OR query, NO objects are added, as I'd expect.
```

## Librarian submit_result answers (the search guidance)

1. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `QuerySet.none()` method is defined in `/testbed/django/db/models/query.py` at lines 919–923.

2. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `QuerySet.union()` method is defined in `/testbed/django/db/models/query.py` at lines 999–1004.

3. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

There is no `CombinedQuerySet` class defined in this repository. The QuerySet-related classes in `/testbed/django/db/models/query.py` are:
- `QuerySet` at line 184
- `EmptyQuerySet` at line 1390
- `RawQuerySet` at line 1400

Combined query functionality (union, intersection, difference) is implemented directly on the `QuerySet` class (lines 999-1019) using the `_combinator_query` method and the `Query.combinator`/`combined_queries` attributes in `/testbed/django/db/models/sql/query.py`.

4. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `none` method body is at `/testbed/django/db/models/query.py` lines 919–923.

5. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

Code from `/testbed/django/db/models/query.py` lines 910–1030 includes the following methods:

- `none()` (lines 919–923) — returns an empty QuerySet
- `all()` (lines 929–934) — returns a copy of the current QuerySet
- `filter()` (lines 936–942) — ANDs args to the existing set
- `exclude()` (lines 944–950) — NOT (args) ANDed to the existing set
- `_filter_or_exclude()` (lines 952–963) — internal filter helper
- `_filter_or_exclude_inplace()` (lines 965–969)
- `complex_filter()` (lines 971–986)
- `_combinator_query()` (lines 988–997) — internal helper for union/intersecti

6. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `_combinator_query` method is at `/testbed/django/db/models/query.py` lines 988–997. Here is its body:

```python
def _combinator_query(self, combinator, *other_qs, all=False):
    # Clone the query to inherit the select list and everything
    clone = self._chain()
    # Clear limits and ordering so they can be reapplied
    clone.query.clear_ordering(True)
    clone.query.clear_limits()
    clone.query.combined_queries = (self.query,) + tuple(qs.query for qs in other_qs)
    clone.query.combinator = combinator
    clone.query.combinator_all = all
    return clone
`

7. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

Both methods are in `/testbed/django/db/models/query.py`:
- `none()` at lines 919–923
- `all()` at lines 929–934

8. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `EmptyQuerySet` class is defined in `/testbed/django/db/models/query.py` at lines 1390–1398. It is a marker class used to check if a queryset was created via `.none()`. It cannot be instantiated directly (raises `TypeError`).

9. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `Query.combinator` attribute is used in `/testbed/django/db/models/sql/query.py` at:

1. **Line 208** — initialized to `None` in `Query.__init__()`
2. **Line 438** — checked in a conditional that determines whether to wrap the query in an `AggregateQuery`

It is also set via `clone.query.combinator = combinator` in the `_combinator_query()` method in `/testbed/django/db/models/query.py` (line 995), which is called by `union()`, `intersection()`, and `difference()`.

10. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

Code from `/testbed/django/db/models/sql/query.py` lines 430–460 shows the logic that wraps a query in an `AggregateQuery` when certain conditions are met — including when `self.combinator` is set (i.e., when the query uses UNION/INTERSECT/EXCEPT). This ensures that set operations are applied in a subquery so that aggregate functions operate on the correct result set.

11. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

`ModelMultipleChoiceField` is defined in `/testbed/django/forms/models.py` at lines 1295–1395. It is a `MultipleChoiceField` whose choices are a model `QuerySet`, extending `ModelChoiceField`.

12. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The body of `ModelMultipleChoiceField` is in `/testbed/django/forms/models.py` lines 1295–1395. Key methods include: `__init__`, `to_python`, `clean`, `_check_values`, `prepare_value`, and `has_changed`.

13. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

`QuerySet.none()` is called in the following locations in `/testbed/django/`:

1. `/testbed/django/db/models/manager.py:204` — `EmptyManager.get_queryset()` returns `super().get_queryset().none()`
2. `/testbed/django/db/models/fields/related_descriptors.py:588` — returns an empty queryset in a related descriptor
3. `/testbed/django/contrib/auth/models.py:189` — `UserManager.empty()` returns `self.none()`
4. `/testbed/django/contrib/auth/backends.py:139` — returns `UserModel._default_manager.none()`
5. `/testbed/django/contrib/contenttypes/forms.py:22` — sets `qs = self.m

14. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

Code from `/testbed/django/forms/models.py` lines 1310–1350 covers the `__init__`, `to_python`, and `clean` methods of `ModelMultipleChoiceField`, plus the beginning of `_check_values`. Key highlights: `clean()` calls `self.queryset.none()` on line 1326 when the field is not required and has no value.

15. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `clean` method of `ModelMultipleChoiceField` is in `/testbed/django/forms/models.py` at lines 1321–1336. It handles validation of the submitted value list, raising `ValidationError` for required/invalid_list cases, then delegates to `_check_values()` to verify each PK against the queryset.

16. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `_check_values` method is defined in `/testbed/django/forms/models.py` at lines 1338–1373. It takes a list of possible PK values and returns a QuerySet of the corresponding objects, raising a `ValidationError` if any value is invalid.

17. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

Code from `/testbed/django/forms/models.py` lines 1320–1380 covers the core validation and processing methods of `ModelMultipleChoiceField`: `clean()`, `_check_values()`, and the start of `prepare_value()`.

18. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `_check_values` method is in `/testbed/django/forms/models.py` at lines 1338–1373. It takes a list of possible PK values and returns a QuerySet of the corresponding objects, raising `ValidationError` for any invalid PKs or values not in the queryset.

19. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `clean` method of `ModelMultipleChoiceField` at lines 1321–1336 in `/testbed/django/forms/models.py`:

```python
def clean(self, value):
    value = self.prepare_value(value)
    if self.required and not value:
        raise ValidationError(self.error_messages['required'], code='required')
    elif not self.required and not value:
        return self.queryset.none()
    if not isinstance(value, (list, tuple)):
        raise ValidationError(
            self.error_messages['invalid_list'],
            code='invalid_list',
        )
    qs = self._check_values(value)
 

20. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `none()` method body is shown below.

21. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `set_empty` method is defined in `/testbed/django/db/models/sql/query.py` at lines 1778–1779. It adds a `NothingNode` to the query's WHERE clause, effectively making the QuerySet return no results.

22. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `set_empty` method body is at `/testbed/django/db/models/sql/query.py` lines 1778–1779. It consists of a single line: `self.where.add(NothingNode(), AND)`.

23. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

`NothingNode` is defined in `/testbed/django/db/models/sql/where.py` at lines 209–214. It is a where clause node that matches nothing — its `as_sql()` method raises `EmptyResultSet`.

24. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `NothingNode` class body was already shown in the previous turn. It is at `/testbed/django/db/models/sql/where.py` lines 209–214.

25. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `is_empty` method is at `/testbed/django/db/models/sql/query.py` lines 1781–1782. It checks if any child in the query's WHERE clause is a `NothingNode`.

26. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

Code from `/testbed/django/db/models/sql/compiler.py` lines 430–450 shows the `get_combinator_sql` method, which handles SQL generation for UNION/INTERSECT/EXCEPT queries. It compiles each combined query and checks database feature support for LIMIT/OFFSET and ORDER BY in compound statements.

27. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `set_empty` method in `/testbed/django/db/models/sql/query.py` (lines 1778–1782) has been updated. The new lines 1780–1782 recursively call `set_empty()` on all combined queries when the query uses a combinator (union/intersection/difference), ensuring that empty QuerySets propagate through compound queries.

28. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

Tests related to `union()` and `none()` are found in:

1. **`/testbed/tests/queries/test_qs_combinators.py`** (lines 1–100+) — Main test file for QuerySet combinators (union, intersection, difference), including tests like:
   - `test_simple_union` (line 19)
   - `test_union_distinct` (line 48)
   - `test_intersection_with_empty_qs` (line 55) — tests union/intersection/difference with `none()`
   - `test_union_with_empty_qs` (line 89)

2. **`/testbed/tests/basic/tests.py`** (lines 306–344) — Tests for `EmptyQuerySet` and `none()`:
   - `test_emptyqs` (line 306)
   - `tes

29. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `test_union_with_empty_qs` test is at `/testbed/tests/queries/test_qs_combinators.py` lines 89–99. It tests that union with an EmptyQuerySet (created via `.none()` or `.filter(pk__in=[])`) returns the non-empty queryset's results, and that union of two empty querysets returns an empty result.

30. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `test_intersection_with_empty_qs` test is at lines 54–64 in `/testbed/tests/queries/test_qs_combinators.py`:

```python
@skipUnlessDBFeature('supports_select_intersection')
def test_intersection_with_empty_qs(self):
    qs1 = Number.objects.all()
    qs2 = Number.objects.none()
    qs3 = Number.objects.filter(pk__in=[])
    self.assertEqual(len(qs1.intersection(qs2)), 0)
    self.assertEqual(len(qs1.intersection(qs3)), 0)
    self.assertEqual(len(qs2.intersection(qs1)), 0)
    self.assertEqual(len(qs3.intersection(qs1)), 0)
    self.assertEqual(len(qs2.intersection(q

31. [submit_result error: view_commands not parseable as JSON list of [path, start, end] — view_commands must be a JSON list]

The `test_union_with_empty_qs` test is at `/testbed/tests/queries/test_qs_combinators.py` lines 89–100. It tests that `union()` with an empty QuerySet (created via `.none()` or `.filter(pk__in=[])`) behaves correctly — the non-empty QuerySet's results are preserved.

## base (PASS) patch

```diff
diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py
index 686e75c979..b26247be82 100644
--- a/django/db/models/sql/compiler.py
+++ b/django/db/models/sql/compiler.py
@@ -502,6 +502,8 @@ class SQLCompiler:
             combinator = self.query.combinator
             features = self.connection.features
             if combinator:
+                if self.query.is_empty():
+                    raise EmptyResultSet
                 if not getattr(features, 'supports_select_{}'.format(combinator)):
                     raise NotSupportedError('{} is not supported on this database backend.'.format(combinator))
                 result, params = self.get_combinator_sql(combinator, self.query.combinator_all)
```

## +Librarian (FAIL) patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
index c913267476..0701ee415e 100644
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1777,6 +1777,9 @@ class Query(BaseExpression):
 
     def set_empty(self):
         self.where.add(NothingNode(), AND)
+        if self.combinator:
+            for query in self.combined_queries:
+                query.set_empty()
 
     def is_empty(self):
         return any(isinstance(c, NothingNode) for c in self.where.children)
```
