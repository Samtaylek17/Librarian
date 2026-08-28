# django__django-11265  (35B-A3B / BOAD)

**Failure mode:** search-crowdout  
**Note:** 16 location dumps; agent edited only the test file

- base (no Librarian) `boad_seed0`: passed=True
- +Librarian `boad_librarian_seed0`: passed=False

**Gold/base edited files:** ['django/db/models/sql/query.py', 'tests/filtered_relation/tests.py']

**+Librarian edited files:** ['tests/filtered_relation/tests.py']

## Issue

```
Using exclude on annotated FilteredRelation doesn't work
Description
	
It looks like using exclude on queryset with annotated FilteredRelation give a FieldError on the annotation name.
For exemple, in Django tests (django/tests/filtered_relation/tests.py) if we change this :
def test_with_join(self):
	self.assertSequenceEqual(
		Author.objects.annotate(
			book_alice=FilteredRelation('book', condition=Q(book__title__iexact='poem by alice')),
		).filter(book_alice__isnull=False),
		[self.author1]
	)
to this
def test_with_join(self):
	self.assertSequenceEqual(
		Author.objects.annotate(
			book_alice=FilteredRelation('book', condition=Q(book__title__iexact='poem by alice')),
		).exclude(book_alice__isnull=False),
		[]
	)
You get the error :
Traceback (most recent call last):
 File "/usr/lib/python3.6/unittest/case.py", line 59, in testPartExecutor
	yield
 File "/usr/lib/python3.6/unittest/case.py", line 605, in run
	testMethod()
 File "/home/lucas/dev/test/django/tests/filtered_relation/tests.py", line 99, in test_with_join_exclude
	).filter(~Q(book_alice__isnull=False)),
 File "/home/lucas/dev/overmind/venvs/release/lib/python3.6/site-packages/django/db/models/query.py", line 844, in filter
	return self._filter_or_exclude(False, *args, **kwargs)
 File "/home/lucas/dev/overmind/venvs/release/lib/python3.6/site-packages/django/db/models/query.py", line 862, in _filter_or_exclude
	clone.query.add_q(Q(*args, **kwargs))
 File "/home/lucas/dev/overmind/venvs/release/lib/python3.6/site-packages/django/db/models/sql/query.py", line 1263, in add_q
	clause, _ = self._add_q(q_object, s
```

## Librarian submit_result answers (the search guidance)

1. The `split_exclude` function is defined in `/testbed/django/db/models/sql/query.py` at lines 1645–1714. It is a method on the `Query` class that constructs a nested subquery for exclude operations against N-to-many relations.

2. Found `FilteredRelation` class in `/testbed/django/db/models/query_utils.py` lines 298-336.

3. The filtered_relation test file is located at `/testbed/tests/filtered_relation/tests.py` (17,336 bytes). The test directory also contains `models.py` with the test models.

4. The `add_filter` method is defined in `/testbed/django/db/models/sql/query.py` at lines 1304–1305. It's a single-line method that delegates to `add_q`.

5. The `Query` class is defined at line 145 in `/testbed/django/db/models/sql/query.py`. The `annotations` property is initialized at line 208 as an empty dict `{}` that maps alias → Annotation Expression. Both are shown in the view below.

6. The `add_q` method is defined in `/testbed/django/db/models/sql/query.py` at lines 1307–1322. It is a preprocessor for the internal `_add_q()` method, responsible for final join promotion. It creates a clause from the q_object, adds it to `self.where` with AND, and demotes any existing inner joins.

7. The `_filtered_relations` attribute is initialized and used in `/testbed/django/db/models/sql/query.py` at these locations:

1. **Initialization** (line 231): `self._filtered_relations = {}` — in the `Query.__init__` method.
2. **Copy during deep copy** (line 330): `obj._filtered_relations = self._filtered_relations.copy()` — in `Query.__deepcopy__`.
3. **Lookup during path resolution** (lines 667-668): Checks if a name is in `_filtered_relations` to get the `relation_name`.
4. **Setting a filtered relation** (line 1388): `self._filtered_relations[filtered_relation.alias] = filtered_relation` — in `add_filtered_relation`.
5. **Path lookup for filtered relations** (lines 1418-1420, 1446): Res

8. The `names_to_path` method is defined in `/testbed/django/db/models/sql/query.py` at lines 1390–1484. It walks a list of names and turns them into PathInfo tuples, handling field resolution, filtered relations, annotations, and inheritance joins.

9. The `trim_start` method is defined in `/testbed/django/db/models/sql/query.py` at lines 2099–2174. It trims joins from the start of the join path for use in generating subquery joins and columns in `split_exclude()`.

10. The `add_filtered_relation` method is defined in `/testbed/django/db/models/sql/query.py` at lines 1377–1388. It assigns an alias to a filtered relation, validates that its condition doesn't reference nested relations, and stores it in `_filtered_relations`.

11. The `build_filter` method is defined in `/testbed/django/db/models/sql/query.py` at lines 1183–1302. It builds a WhereNode for a single filter clause, handling join resolution, lookup building, and delegating to `split_exclude()` when a MultiJoin is encountered.

12. There is no `setup` method in `/testbed/django/db/models/sql/query.py`. The closest match is `setup_joins` at lines 1486–1586, which computes the necessary table joins for passage through fields given in 'names'.

13. FilteredRelation is processed during annotation in the Query class at these key locations in `/testbed/django/db/models/sql/query.py`:

1. **add_filtered_relation** (lines 1377-1388): Stores the FilteredRelation in `self._filtered_relations` dict after validating its condition doesn't support nested relations.

2. **build_filtered_relation_q** (lines 1355-1375): Builds a WhereNode for a FilteredRelation object, calling `build_filter` with `reuse_with_filtered_relation=True`.

3. **names_to_path** (lines 1390-1484): Resolves field paths, checking `self._filtered_relations` at line 1418-1420 to handle filtered relation lookups when `pos == 0`.

4. **setup_joins** (lines 1486-1586): Processes j

14. The `join` method is defined in `/testbed/django/db/models/sql/query.py` at lines 927–973. It returns an alias for a join, either reusing an existing alias or creating a new one. It handles the `reuse_with_filtered_relation` parameter for FilteredRelation instances.

15. The `table_alias` method is defined in `/testbed/django/db/models/sql/query.py` at lines 727–750. It returns a table alias for a given table_name, creating a new one if `create=True` or reusing an existing one. When a `filtered_relation` is provided, it uses `filtered_relation.alias` as the alias for the first occurrence of the table.

16. The `get_path_info` method used by `ForeignKey` is inherited from `ForeignObject` and is defined in `/testbed/django/db/models/fields/related.py` at lines 698–710. It returns a PathInfo from this field to the related model, including the `filtered_relation` parameter.

17. The FilteredRelation's condition is compiled into SQL in two places:

1. **Join.as_sql** in `/testbed/django/db/models/sql/datastructures.py` (lines 89-93): When generating the JOIN ON clause, it compiles `self.filtered_relation` via `compiler.compile(self.filtered_relation)`.

2. **FilteredRelation.as_sql** in `/testbed/django/db/models/query_utils.py` (lines 332-336): This method resolves the condition by calling `query.build_filtered_relation_q(self.condition, reuse=set(self.path))` and then compiles the resulting WhereNode.

The flow is: Join.as_sql → compiler.compile(FilteredRelation) → FilteredRelation.as_sql → query.build_filtered_relation_q → compiler.compile(WhereNode).

18. The FilteredRelation's condition is resolved when building the WHERE clause in the subquery in `split_exclude` at `/testbed/django/db/models/sql/query.py` lines 1676-1685:

```python
# Apply the filtered_relation's condition to the WHERE clause.
# The condition was previously in the ON clause of the join, but after
# trim_start converts the join to a BaseTable, we need to add it to
# the WHERE clause.
for alias, filtered_relation in query._filtered_relations.items():
    if alias in query.alias_map:
        where = query.build_filtered_relation_q(
            filtered_relation.condition, reuse=set(filtered_relation.path))
        query.where.add(where, AND)
        break
```

The flow is:
1.

## base (PASS) patch

```diff
diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py
index d69c24419b..316e50aa51 100644
--- a/django/db/models/sql/query.py
+++ b/django/db/models/sql/query.py
@@ -1666,10 +1666,25 @@ class Query(BaseExpression):
             filter_expr = (filter_lhs, OuterRef(filter_rhs.name))
         # Generate the inner query.
         query = Query(self.model)
+        query._filtered_relations = self._filtered_relations.copy()
         query.add_filter(filter_expr)
         query.clear_ordering(True)
         # Try to have as simple as possible subquery -> trim leading joins from
         # the subquery.
+        # Before trimming, we need to add the filtered relations' conditions
+        # to the WHERE clause since they will be lost when the Join is converted
+        # to a BaseTable during trimming. However, we must not add the condition
+        # for LEFT OUTER JOINs as it would filter out rows that have no matching
+        # related object.
+        for alias, join in query.alias_map.items():
+            if (hasattr(join, 'filtered_relation') and
+                    join.filtered_relation and
+                    join.join_type != LOUTER):
+                filtered_relation = join.filtered_relation
+                where = query.build_filtered_relation_q(
+                    filtered_relation.condition, reuse=set(filtered_relation.path),
+                )
+                query.where.add(where, AND)
         trimmed_prefix, contains_louter = query.trim_start(names_with_path)
 
         # Add extra check to make sure the selected field will not be null
diff --git a/tests/filtered_relation/tests.py b/tests/filtered_relation/tests.py
index 5e85dff16c..a60de7455b 100644
--- a/tests/filtered_relation/tests.py
+++ b/tests/filtered_relation/tests.py
@@ -147,6 +147,14 @@ class FilteredRelationTests(TestCase):
             [self.author2]
         )
 
+    def test_exclude_with_filtered_relation(self):
+        self.assertSequenceEqual(
+            Author.objects.annotate(
+                book_alice=FilteredRelation('book', condition=Q(book__title__iexact='poem by alice')),
+            ).exclude(book_alice__isnull=False),
+            [self.author2]
+        )
+
     def test_with_m2m(self):
         qs = Author.objects.annotate(
             favorite_books_written_by_jane=FilteredRelation(
```

## +Librarian (FAIL) patch

```diff
diff --git a/tests/filtered_relation/tests.py b/tests/filtered_relation/tests.py
index 5e85dff16c..a60de7455b 100644
--- a/tests/filtered_relation/tests.py
+++ b/tests/filtered_relation/tests.py
@@ -147,6 +147,14 @@ class FilteredRelationTests(TestCase):
             [self.author2]
         )
 
+    def test_exclude_with_filtered_relation(self):
+        self.assertSequenceEqual(
+            Author.objects.annotate(
+                book_alice=FilteredRelation('book', condition=Q(book__title__iexact='poem by alice')),
+            ).exclude(book_alice__isnull=False),
+            [self.author2]
+        )
+
     def test_with_m2m(self):
         qs = Author.objects.annotate(
             favorite_books_written_by_jane=FilteredRelation(
```
