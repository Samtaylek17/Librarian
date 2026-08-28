# django__django-11149  (35B-A3B / HyperAgent)

**Failure mode:** stale-negative  
**Note:** lib claimed 'NO tests for view-only m2m'; gold adds that test

- base (no Librarian) `hyperagent_seed0`: passed=True
- +Librarian `hyperagent_librarian_seed0`: passed=False

**Gold/base edited files:** ['django/contrib/admin/options.py', 'tests/test_settings.py']

**+Librarian edited files:** ['django/contrib/admin/options.py']

## Issue

```
Admin inlines for auto-created ManyToManyFields are editable if the user only has the view permission
Description
	
From https://code.djangoproject.com/ticket/8060#comment:34
Replying to Will Gordon:
This seems to have regressed in (at least) 2.1. I have 2 view only permissions. I have a ManyToManyField represented in my main model as a TabularInline. But, my user with view only permissions can now add or remove these items at will!
I am having the same issue, so I assume this is a bug. I did not find Will had created a separate ticket.
models.py:
class Photo(models.Model):
	pass
class Report(models.Model):
	photos = models.ManyToManyField(Photo)
admin.py:
		class ReportPhotoInlineModelAdmin(admin.TabularInline):
			model = Report.photos.through
			show_change_link = True
```

## Librarian submit_result answers (the search guidance)

1. The admin inline permission checks for ManyToManyField through models are in:

**Primary location: `/testbed/django/contrib/admin/options.py` lines 2114-2154**

The `InlineModelAdmin` class (defined at line 1960) handles permission checks for inlines. For auto-created through models (ManyToMany intermediary tables), the key methods are:

- `has_add_permission` (lines 2114-2121)
- `has_change_permission` (lines 2123-2130)  
- `has_delete_permission` (lines 2132-2139)
- `has_view_permission` (lines 2141-2154)

All four methods check `self.opts.auto_created` to detect if the inline model is an auto-created through table. When `auto_created` is True, permissions are delegated to the related/targ

2. The `get_formset` method in `InlineModelAdmin` is at `/testbed/django/contrib/admin/options.py` lines 2016-2103. It sets the permission flags at lines 2032, 2048-2049:

- Line 2032: `can_delete = self.can_delete and self.has_delete_permission(request, obj)`
- Line 2048: `can_change = self.has_change_permission(request, obj) if request else True`
- Line 2049: `can_add = self.has_add_permission(request, obj) if request else True`

These flags are passed to the formset factory and used in the dynamically-created `DeleteProtectedModelForm` (lines 2051-2098) to control `has_changed()` behavior.

3. The `InlineModelAdmin` class is defined at:

**File:** `/testbed/django/contrib/admin/options.py`
**Lines:** 1960-2155

The class declaration starts at line 1960 (`class InlineModelAdmin(BaseModelAdmin):`) and includes all methods through line 2155 (`has_view_permission`). The subclasses `StackedInline` (line 2157) and `TabularInline` (line 2161) follow immediately after.

4. The `ModelAdmin` permission methods are in `/testbed/django/contrib/admin/options.py` lines 463-522:

- `has_add_permission` (lines 463-470) - checks for `add` codename permission
- `has_change_permission` (lines 472-485) - checks for `change` codename permission
- `has_delete_permission` (lines 487-500) - checks for `delete` codename permission
- `has_view_permission` (lines 502-519) - checks for `view` OR `change` codename permission
- `has_view_or_change_permission` (lines 521-522) - convenience method combining view and change

5. Test files for inline permissions (including ManyToMany through table permissions):

1. **`/testbed/tests/admin_inlines/tests.py`** - Contains the `TestInlinePermissions` class (line 568) with tests specifically for ManyToMany auto-created through table permissions:
   - `test_inline_add_m2m_noperm` (line 611)
   - `test_inline_change_m2m_noperm` (line 625)
   - `test_inline_add_m2m_add_perm` (line 639)
   - `test_inline_change_m2m_add_perm` (line 658)
   - `test_inline_change_m2m_change_perm` (line 668)

2. **`/testbed/tests/admin_inlines/admin.py`** - Contains the `BookInline` (line 18-19) that uses `Author.books.through` (auto-created ManyToMany through model) as the inline model, registe

6. There are NO specific tests for view-only permissions on ManyToMany inlines in the admin_inlines tests. The `TestInlinePermissions` class in `/testbed/tests/admin_inlines/tests.py` (lines 568-750) only tests add, change, and delete permissions for ManyToMany through inlines.

However, there are view permission tests for ForeignKey inlines in `/testbed/tests/admin_views/tests.py`:
- `test_change_view_with_view_and_add_inlines` (line 1985) - tests view+add on ForeignKey inlines
- `test_change_view_with_view_and_delete_inlines` (line 2019) - tests view+delete on ForeignKey inlines

The `InlineModelAdmin.has_view_permission` method (lines 2141-2154 in `/testbed/django/contrib/admin/options.py`) 

7. Confirmed: There are NO tests for view-only permissions on ManyToMany inlines in `/testbed/tests/admin_inlines/tests.py`.

The `TestInlinePermissions` class (lines 568-750) has these test methods:
- `test_inline_add_m2m_noperm` - tests no permission scenario
- `test_inline_add_fk_noperm` - tests no permission scenario (FK)
- `test_inline_change_m2m_noperm` - tests no permission scenario
- `test_inline_change_fk_noperm` - tests no permission scenario (FK)
- `test_inline_add_m2m_add_perm` - tests add permission only
- `test_inline_add_fk_add_perm` - tests add permission only (FK)
- `test_inline_change_m2m_add_perm` - tests add permission only
- `test_inline_change_m2m_change_perm` - tests chan

8. The inline formset visibility is determined in **`/testbed/django/contrib/admin/options.py`** at the `get_inline_instances` method (lines 583-596):

```python
def get_inline_instances(self, request, obj=None):
    inline_instances = []
    for inline_class in self.inlines:
        inline = inline_class(self.model, self.admin_site)
        if request:
            if not (inline.has_view_or_change_permission(request, obj) or
                    inline.has_add_permission(request, obj) or
                    inline.has_delete_permission(request, obj)):
                continue
            if not inline.has_add_permission(request, obj):
                inline.max_num = 0
        inline_instances.

9. The **tabular.html template** (`/testbed/django/contrib/admin/templates/admin/edit_inline/tabular.html`) handles the "Add another" functionality via JavaScript, not directly in the template:

**Key elements:**

1. **Line 28** - The empty form template row:
   ```html
   <tr class="form-row ... {% if forloop.last and inline_admin_formset.has_add_permission %} empty-form{% endif %}"
   ```
   The `empty-form` class is only added to the last row if `has_add_permission` is True.

2. **Line 4** - The `data-inline-formset` attribute passes configuration to JavaScript:
   ```html
   data-inline-formset="{{ inline_admin_formset.inline_formset_data }}"
   ```
   This includes the `addText` option (fr

10. The `InlineModelAdmin` permission methods are in **`/testbed/django/contrib/admin/options.py`** lines 2114-2154:

- `has_add_permission` (lines 2114-2121) - checks `self.opts.auto_created` and delegates to `has_view_permission` for auto-created through tables
- `has_change_permission` (lines 2123-2130) - same auto_created check, delegates to `has_view_permission`
- `has_delete_permission` (lines 2132-2139) - same auto_created check, delegates to `has_view_permission`
- `has_view_permission` (lines 2141-2154) - for auto_created through tables, finds the target model and checks for `view` OR `change` permission on it

11. The `get_inline_instances` method is in the `ModelAdmin` class at **`/testbed/django/contrib/admin/options.py`** lines 583-596. This was already shown in the previous response.

12. Tests related to `has_view_or_change_permission` and inline permissions:

1. **`/testbed/tests/admin_views/test_autocomplete_view.py`** - `test_has_view_or_change_permission_required` (line 72) - Tests that the autocomplete view requires view or change permission on the related model.

2. **`/testbed/tests/admin_views/tests.py`** - `test_add_view_with_view_only_inlines` (line 1758) - Tests that `InlineModelAdmin.has_change_permission` receives `obj=None` during add view, mocking the method to verify behavior.

3. **`/testbed/tests/admin_views/tests.py`** - `test_simple_inline_permissions` (line 4075) - Tests inline permissions for ForeignKey inlines (view-only prevents changes, add/change pe

## base (PASS) patch

```diff
diff --git a/django/contrib/admin/options.py b/django/contrib/admin/options.py
index 880a2cc17b..f0a40970dc 100644
--- a/django/contrib/admin/options.py
+++ b/django/contrib/admin/options.py
@@ -2115,27 +2115,42 @@ class InlineModelAdmin(BaseModelAdmin):
         if self.opts.auto_created:
             # We're checking the rights to an auto-created intermediate model,
             # which doesn't have its own individual permissions. The user needs
-            # to have the view permission for the related model in order to
-            # be able to do anything with the intermediate model.
-            return self.has_view_permission(request, obj)
+            # to have the change permission for the related model in order to
+            # be able to add items through the intermediate model.
+            opts = self.opts
+            for field in opts.fields:
+                if field.remote_field and field.remote_field.model != self.parent_model:
+                    opts = field.remote_field.model._meta
+                    break
+            return request.user.has_perm('%s.%s' % (opts.app_label, get_permission_codename('change', opts)))
         return super().has_add_permission(request)
 
     def has_change_permission(self, request, obj=None):
         if self.opts.auto_created:
             # We're checking the rights to an auto-created intermediate model,
             # which doesn't have its own individual permissions. The user needs
-            # to have the view permission for the related model in order to
-            # be able to do anything with the intermediate model.
-            return self.has_view_permission(request, obj)
+            # to have the change permission for the related model in order to
+            # be able to change items through the intermediate model.
+            opts = self.opts
+            for field in opts.fields:
+                if field.remote_field and field.remote_field.model != self.parent_model:
+                    opts = field.remote_field.model._meta
+                    break
+            return request.user.has_perm('%s.%s' % (opts.app_label, get_permission_codename('change', opts)))
         return super().has_change_permission(request)
 
     def has_delete_permission(self, request, obj=None):
         if self.opts.auto_created:
             # We're checking the rights to an auto-created intermediate model,
             # which doesn't have its own individual permissions. The user needs
-            # to have the view permission for the related model in order to
-            # be able to do anything with the intermediate model.
-            return self.has_view_permission(request, obj)
+            # to have the change permission for the related model in order to
+            # be able to delete items through the intermediate model.
+            opts = self.opts
+            for field in opts.fields:
+                if field.remote_field and field.remote_field.model != self.parent_model:
+                    opts = field.remote_field.model._meta
+                    break
+            return request.user.has_perm('%s.%s' % (opts.app_label, get_permission_codename('change', opts)))
         return super().has_delete_permission(request, obj)
 
     def has_view_permission(self, request, obj=None):
diff --git a/tests/test_settings.py b/tests/test_settings.py
new file mode 100644
index 0000000000..9d54d9f69f
--- /dev/null
+++ b/tests/test_settings.py
@@ -0,0 +1,15 @@
+DATABASES = {
+ 
```

## +Librarian (FAIL) patch

```diff
diff --git a/django/contrib/admin/options.py b/django/contrib/admin/options.py
index 880a2cc17b..bccf31fd1a 100644
--- a/django/contrib/admin/options.py
+++ b/django/contrib/admin/options.py
@@ -2115,27 +2115,27 @@ class InlineModelAdmin(BaseModelAdmin):
         if self.opts.auto_created:
             # We're checking the rights to an auto-created intermediate model,
             # which doesn't have its own individual permissions. The user needs
-            # to have the view permission for the related model in order to
-            # be able to do anything with the intermediate model.
-            return self.has_view_permission(request, obj)
+            # to have the change permission for the related model in order to
+            # add a relationship.
+            return self._has_related_model_perm(request, 'change')
         return super().has_add_permission(request)
 
     def has_change_permission(self, request, obj=None):
         if self.opts.auto_created:
             # We're checking the rights to an auto-created intermediate model,
             # which doesn't have its own individual permissions. The user needs
-            # to have the view permission for the related model in order to
-            # be able to do anything with the intermediate model.
-            return self.has_view_permission(request, obj)
+            # to have the change permission for the related model in order to
+            # change a relationship.
+            return self._has_related_model_perm(request, 'change')
         return super().has_change_permission(request)
 
     def has_delete_permission(self, request, obj=None):
         if self.opts.auto_created:
             # We're checking the rights to an auto-created intermediate model,
             # which doesn't have its own individual permissions. The user needs
-            # to have the view permission for the related model in order to
-            # be able to do anything with the intermediate model.
-            return self.has_view_permission(request, obj)
+            # to have the change permission for the related model in order to
+            # delete a relationship.
+            return self._has_related_model_perm(request, 'change')
         return super().has_delete_permission(request, obj)
 
     def has_view_permission(self, request, obj=None):
@@ -2147,12 +2147,33 @@ class InlineModelAdmin(BaseModelAdmin):
                 if field.remote_field and field.remote_field.model != self.parent_model:
                     opts = field.remote_field.model._meta
                     break
-            return (
-                request.user.has_perm('%s.%s' % (opts.app_label, get_permission_codename('view', opts))) or
-                request.user.has_perm('%s.%s' % (opts.app_label, get_permission_codename('change', opts)))
+            return request.user.has_perm(
+                '%s.%s' % (opts.app_label, get_permission_codename('view', opts))
             )
         return super().has_view_permission(request)
 
+    def _has_related_model_perm(self, request, perm):
+        """
+        Check if the user has the given permission on the related model.
+        Used for auto-created intermediate models (ManyToMany through tables).
+        """
+        opts = self.opts
+        for field in opts.fields:
+            if field.remote_field and field.remote_field.model != self.parent_model:
+                opts = field.remote_field.model._meta
+                break
+        return 
```
