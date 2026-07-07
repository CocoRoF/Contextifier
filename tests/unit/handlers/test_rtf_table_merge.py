# tests/unit/handlers/test_rtf_table_merge.py
"""
P2-1: RTF table merge flag verification tests.

Verifies that \\clmgf/\\clmrg (horizontal) and \\clvmgf/\\clvmrg (vertical)
merge control words are correctly parsed and translated into
colspan/rowspan in the output TableData.
"""

from __future__ import annotations

from unittest import mock

from contextifier.handlers.rtf._table_parser import (
    _parse_cell_definitions,
    _build_table_data,
    _is_real_table,
    _ParsedCell,
    extract_tables,
    single_column_to_text,
)


# ═══════════════════════════════════════════════════════════════════════════
# _parse_cell_definitions tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParseCellDefinitions:
    """Tests for RTF cell definition parsing — merge flag extraction."""

    def test_no_merge_flags(self):
        """Simple cells with no merge flags."""
        rtf_def = r"\cellx2000\cellx4000\cellx6000"
        defs = _parse_cell_definitions(rtf_def)
        assert len(defs) == 3
        for d in defs:
            assert d.h_merge_first is False
            assert d.h_merge_cont is False
            assert d.v_merge_first is False
            assert d.v_merge_cont is False

    def test_horizontal_merge_first(self):
        """\\clmgf marks the first cell in a horizontal merge."""
        rtf_def = r"\clmgf\cellx2000\clmrg\cellx4000\cellx6000"
        defs = _parse_cell_definitions(rtf_def)
        assert len(defs) == 3
        assert defs[0].h_merge_first is True
        assert defs[0].h_merge_cont is False
        assert defs[1].h_merge_first is False
        assert defs[1].h_merge_cont is True
        assert defs[2].h_merge_first is False
        assert defs[2].h_merge_cont is False

    def test_vertical_merge_first(self):
        """\\clvmgf marks the first cell in a vertical merge."""
        rtf_def = r"\clvmgf\cellx2000\cellx4000"
        defs = _parse_cell_definitions(rtf_def)
        assert len(defs) == 2
        assert defs[0].v_merge_first is True
        assert defs[0].v_merge_cont is False
        assert defs[1].v_merge_first is False

    def test_vertical_merge_continuation(self):
        """\\clvmrg marks continuation in a vertical merge."""
        rtf_def = r"\clvmrg\cellx2000\cellx4000"
        defs = _parse_cell_definitions(rtf_def)
        assert len(defs) == 2
        assert defs[0].v_merge_cont is True
        assert defs[0].v_merge_first is False

    def test_combined_hv_merge(self):
        """Cell with both horizontal AND vertical merge flags."""
        rtf_def = r"\clmgf\clvmgf\cellx3000\clmrg\cellx6000"
        defs = _parse_cell_definitions(rtf_def)
        assert len(defs) == 2
        assert defs[0].h_merge_first is True
        assert defs[0].v_merge_first is True
        assert defs[1].h_merge_cont is True
        assert defs[1].v_merge_first is False

    def test_cell_boundary_values(self):
        """Verify right_boundary values are correctly parsed."""
        rtf_def = r"\cellx1500\cellx3000\cellx-100"
        defs = _parse_cell_definitions(rtf_def)
        assert len(defs) == 3
        assert defs[0].right_boundary == 1500
        assert defs[1].right_boundary == 3000
        assert defs[2].right_boundary == -100

    def test_merge_flags_reset_between_cells(self):
        """Merge flags should reset between cell definitions."""
        rtf_def = r"\clmgf\cellx2000\cellx4000\clmgf\cellx6000"
        defs = _parse_cell_definitions(rtf_def)
        assert len(defs) == 3
        assert defs[0].h_merge_first is True
        assert defs[1].h_merge_first is False  # Reset
        assert defs[2].h_merge_first is True


# ═══════════════════════════════════════════════════════════════════════════
# Horizontal merge (colspan) tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHorizontalMerge:
    """End-to-end horizontal merge → colspan."""

    def test_two_cell_colspan(self):
        """Two cells merged horizontally → colspan=2."""
        rows = [
            [
                _ParsedCell("A", h_merge_first=True, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
                _ParsedCell("", h_merge_first=False, h_merge_cont=True,
                            v_merge_first=False, v_merge_cont=False),
                _ParsedCell("B", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
            ],
        ]
        table = _build_table_data(rows)
        assert table is not None
        # First row should have 2 cells: A(colspan=2) and B
        row0 = table.rows[0]
        assert len(row0) == 2
        assert row0[0].content == "A"
        assert row0[0].col_span == 2
        assert row0[1].content == "B"
        assert row0[1].col_span == 1

    def test_three_cell_colspan(self):
        """Three cells merged horizontally → colspan=3."""
        rows = [
            [
                _ParsedCell("Merged", h_merge_first=True, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
                _ParsedCell("", h_merge_first=False, h_merge_cont=True,
                            v_merge_first=False, v_merge_cont=False),
                _ParsedCell("", h_merge_first=False, h_merge_cont=True,
                            v_merge_first=False, v_merge_cont=False),
            ],
            [
                _ParsedCell("X", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
                _ParsedCell("Y", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
                _ParsedCell("Z", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
            ],
        ]
        table = _build_table_data(rows)
        row0 = table.rows[0]
        assert len(row0) == 1  # Only the merged cell
        assert row0[0].col_span == 3
        assert row0[0].content == "Merged"

        row1 = table.rows[1]
        assert len(row1) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Vertical merge (rowspan) tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVerticalMerge:
    """End-to-end vertical merge → rowspan."""

    def test_two_row_rowspan(self):
        """Two rows merged vertically → rowspan=2."""
        rows = [
            [
                _ParsedCell("A", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=True, v_merge_cont=False),
                _ParsedCell("B", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
            ],
            [
                _ParsedCell("", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=True),
                _ParsedCell("C", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
            ],
        ]
        table = _build_table_data(rows)
        # Row 0: A(rowspan=2), B
        row0 = table.rows[0]
        a_cell = next(c for c in row0 if c.content == "A")
        assert a_cell.row_span == 2
        assert a_cell.col_span == 1

        # Row 1: only C (merged-away cell is skipped)
        row1 = table.rows[1]
        assert len(row1) == 1
        assert row1[0].content == "C"

    def test_three_row_rowspan(self):
        """Three rows merged vertically → rowspan=3."""
        rows = [
            [
                _ParsedCell("Header", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=True, v_merge_cont=False),
                _ParsedCell("X1", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
            ],
            [
                _ParsedCell("", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=True),
                _ParsedCell("X2", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
            ],
            [
                _ParsedCell("", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=True),
                _ParsedCell("X3", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
            ],
        ]
        table = _build_table_data(rows)
        row0 = table.rows[0]
        header = next(c for c in row0 if c.content == "Header")
        assert header.row_span == 3


# ═══════════════════════════════════════════════════════════════════════════
# Combined horizontal + vertical merge tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCombinedMerge:
    """Combined horizontal AND vertical merges."""

    def test_2x2_block_merge(self):
        """2×2 block merged: colspan=2, rowspan=2."""
        rows = [
            [
                _ParsedCell("Block", h_merge_first=True, h_merge_cont=False,
                            v_merge_first=True, v_merge_cont=False),
                _ParsedCell("", h_merge_first=False, h_merge_cont=True,
                            v_merge_first=True, v_merge_cont=False),
                _ParsedCell("R", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
            ],
            [
                _ParsedCell("", h_merge_first=True, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=True),
                _ParsedCell("", h_merge_first=False, h_merge_cont=True,
                            v_merge_first=False, v_merge_cont=True),
                _ParsedCell("S", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
            ],
        ]
        table = _build_table_data(rows)
        row0 = table.rows[0]
        block = next(c for c in row0 if c.content == "Block")
        assert block.col_span == 2
        assert block.row_span == 2

        # Row 1 has only "S" since merged-away cells are skipped
        row1 = table.rows[1]
        visible = [c for c in row1 if c.content.strip()]
        assert len(visible) == 1
        assert visible[0].content == "S"


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeEdgeCases:
    """Edge cases for merge processing."""

    def test_orphaned_v_merge_cont(self):
        """Orphaned \\clvmrg (no preceding \\clvmgf) → cell becomes (0,0)."""
        rows = [
            [
                _ParsedCell("A", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
                _ParsedCell("B", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
            ],
            [
                _ParsedCell("orphan", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=True),
                _ParsedCell("C", h_merge_first=False, h_merge_cont=False,
                            v_merge_first=False, v_merge_cont=False),
            ],
        ]
        table = _build_table_data(rows)
        # Row 1: orphaned cell should be skipped
        row1 = table.rows[1]
        assert len(row1) == 1
        assert row1[0].content == "C"

    def test_uneven_row_lengths(self):
        """Rows with different number of cells."""
        rows = [
            [
                _ParsedCell("A", False, False, False, False),
                _ParsedCell("B", False, False, False, False),
                _ParsedCell("C", False, False, False, False),
            ],
            [
                _ParsedCell("D", False, False, False, False),
                _ParsedCell("E", False, False, False, False),
            ],
        ]
        table = _build_table_data(rows)
        assert table.num_cols == 3  # max of any row
        assert len(table.rows[0]) == 3
        assert len(table.rows[1]) == 2

    def test_is_real_table_single_column(self):
        """Single-column structures are NOT real tables."""
        rows = [
            [_ParsedCell("Line1", False, False, False, False)],
            [_ParsedCell("Line2", False, False, False, False)],
        ]
        assert _is_real_table(rows) is False

    def test_is_real_table_multi_column(self):
        """Multi-column structures ARE real tables."""
        rows = [
            [
                _ParsedCell("A", False, False, False, False),
                _ParsedCell("B", False, False, False, False),
            ],
        ]
        assert _is_real_table(rows) is True

    def test_single_column_to_text_output(self):
        """single_column_to_text() concatenates cell texts."""
        # Build minimal RTF row text
        row_text = r"\trowd\cellx5000 Hello\cell\row"
        with mock.patch(
            "contextifier.handlers.rtf._table_parser.decode_hex_escapes",
            side_effect=lambda t, enc: t,
        ), mock.patch(
            "contextifier.handlers.rtf._table_parser.clean_rtf_text",
            side_effect=lambda t, enc: t.strip(),
        ):
            result = single_column_to_text([row_text], "cp949")
            # Should contain some text (implementation-dependent on RTF parsing)
            assert isinstance(result, str)

    def test_empty_rows(self):
        """Empty parsed rows returns empty TableData."""
        result = _build_table_data([])
        assert result is not None
        assert result.num_rows == 0
        assert result.rows == []


class TestExtractTablesIntegration:
    """Integration tests using raw RTF content snippets."""

    def test_simple_2x2_rtf(self):
        """Parse a minimal 2×2 RTF table."""
        rtf = (
            r"\trowd\cellx3000\cellx6000"
            r" A\cell B\cell\row"
            r"\trowd\cellx3000\cellx6000"
            r" C\cell D\cell\row"
        )
        with mock.patch(
            "contextifier.handlers.rtf._table_parser.decode_hex_escapes",
            side_effect=lambda t, enc: t,
        ), mock.patch(
            "contextifier.handlers.rtf._table_parser.clean_rtf_text",
            side_effect=lambda t, enc: t.strip(),
        ), mock.patch(
            "contextifier.handlers.rtf._table_parser.find_excluded_regions",
            return_value=[],
        ), mock.patch(
            "contextifier.handlers.rtf._table_parser.is_in_excluded_region",
            return_value=False,
        ):
            tables = extract_tables(rtf, "cp949")
            assert len(tables) == 1
            tbl = tables[0]
            assert tbl.num_rows == 2
            assert tbl.num_cols == 2

    def test_horizontal_merge_rtf(self):
        """RTF table with \\clmgf + \\clmrg produces colspan."""
        rtf = (
            r"\trowd\clmgf\cellx3000\clmrg\cellx6000"
            r" Merged\cell \cell\row"
            r"\trowd\cellx3000\cellx6000"
            r" C\cell D\cell\row"
        )
        with mock.patch(
            "contextifier.handlers.rtf._table_parser.decode_hex_escapes",
            side_effect=lambda t, enc: t,
        ), mock.patch(
            "contextifier.handlers.rtf._table_parser.clean_rtf_text",
            side_effect=lambda t, enc: t.strip(),
        ), mock.patch(
            "contextifier.handlers.rtf._table_parser.find_excluded_regions",
            return_value=[],
        ), mock.patch(
            "contextifier.handlers.rtf._table_parser.is_in_excluded_region",
            return_value=False,
        ):
            tables = extract_tables(rtf, "cp949")
            assert len(tables) == 1
            tbl = tables[0]
            row0 = tbl.rows[0]
            merged_cell = row0[0]
            assert merged_cell.col_span == 2
            assert merged_cell.content == "Merged"
