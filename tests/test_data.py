"""Tests for SWE-bench data loading and parsing."""

import pytest
from envaudit.data.swebench import (
    extract_test_code_from_diff,
    count_test_functions,
    count_assertions,
    categorize_assertions,
    extract_full_diff_context,
)


SAMPLE_DIFF = """diff --git a/tests/test_math.py b/tests/test_math.py
--- a/tests/test_math.py
+++ b/tests/test_math.py
@@ -1,5 +1,20 @@
 import unittest
+from mymodule import add, multiply

 class TestMath(unittest.TestCase):
+    def test_add_basic(self):
+        self.assertEqual(add(1, 2), 3)
+        self.assertEqual(add(0, 0), 0)
+
+    def test_add_negative(self):
+        self.assertEqual(add(-1, 1), 0)
+        self.assertTrue(add(-5, -3) < 0)
+
+    def test_multiply(self):
+        self.assertEqual(multiply(2, 3), 6)
+        self.assertIn(multiply(0, 5), [0])
+
+    def test_edge_empty(self):
+        with self.assertRaises(TypeError):
+            add(None, 1)
"""


class TestExtractTestCode:
    def test_extracts_added_lines(self):
        code = extract_test_code_from_diff(SAMPLE_DIFF)
        assert "def test_add_basic" in code
        assert "assertEqual(add(1, 2), 3)" in code

    def test_excludes_removed_lines(self):
        code = extract_test_code_from_diff(SAMPLE_DIFF)
        # No removed lines in this diff
        assert "diff --git" not in code

    def test_empty_diff(self):
        assert extract_test_code_from_diff("") == ""
        assert extract_test_code_from_diff(None) == ""

    def test_includes_context(self):
        code = extract_test_code_from_diff(SAMPLE_DIFF)
        assert "import unittest" in code


class TestCountFunctions:
    def test_counts_test_functions(self):
        code = extract_test_code_from_diff(SAMPLE_DIFF)
        assert count_test_functions(code) == 4

    def test_zero_for_empty(self):
        assert count_test_functions("") == 0


class TestCountAssertions:
    def test_counts_all_assertions(self):
        code = extract_test_code_from_diff(SAMPLE_DIFF)
        n = count_assertions(code)
        assert n >= 6  # 3 assertEqual + 1 assertTrue + 1 assertIn + 1 assertRaises


class TestCategorizeAssertions:
    def test_categorizes_correctly(self):
        code = extract_test_code_from_diff(SAMPLE_DIFF)
        cats = categorize_assertions(code)
        assert cats["assertEqual"] == 4  # add(1,2)=3, add(0,0)=0, add(-1,1)=0, multiply(2,3)=6
        assert cats["assertTrue"] == 1
        assert cats["assertIn"] == 1
        assert cats["assertRaises"] == 1


class TestFullDiffContext:
    def test_removes_git_header(self):
        ctx = extract_full_diff_context(SAMPLE_DIFF)
        assert "diff --git" not in ctx
        assert "index " not in ctx

    def test_keeps_hunks(self):
        ctx = extract_full_diff_context(SAMPLE_DIFF)
        assert "@@" in ctx
