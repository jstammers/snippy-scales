"""Unit tests for snippy_scales.data.universe (S&P 500 historical membership).

Parsing is tested against small hand-written HTML snippets that mirror the
real Wikipedia table structure (id, column order, two-row header on the
changes table) rather than a full cached page — see the module docstring
for why the real page structure matters (it moved once already, in 2026).
network access (fetch_current_constituents/fetch_membership_changes) is
never exercised here.
"""

from __future__ import annotations

import datetime as dt

import pytest

from snippy_scales.data.universe import (
    MembershipChange,
    parse_changes_table,
    parse_constituents_table,
    sp500_ever_members,
)

_CONSTITUENTS_HTML = """
<html><body>
<table id="constituents" class="wikitable sortable">
  <tbody>
    <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
    <tr><td>MMM</td><td>3M</td><td>Industrials</td></tr>
    <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td></tr>
    <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>
  </tbody>
</table>
</body></html>
"""

# Mirrors the real page's two header rows (Effective Date/Added/Removed/Reason/Refs,
# then Ticker/Security/Ticker/Security) and mixed 6/7-cell data rows.
_CHANGES_HTML = """
<html><body>
<table id="changes" class="wikitable sortable">
  <tbody>
    <tr><th>Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th>
        <th>Reason</th><th>Refs</th></tr>
    <tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
    <tr>
      <td>August 18, 2026</td><td>RDDT</td><td>Reddit</td>
      <td>AVB</td><td>AvalonBay Communities</td><td>Merger.</td><td>[2]</td>
    </tr>
    <tr>
      <td>June 30, 2026</td><td></td><td></td>
      <td>CAG</td><td>Conagra Brands</td><td>Market capitalization changes.</td><td>[4]</td>
    </tr>
    <tr>
      <td>September 20, 2021</td><td>BRO</td><td>Brown &amp; Brown</td>
      <td>NOV</td><td>NOV Inc.</td><td>Market capitalization changes.</td><td>[61]</td>
    </tr>
    <tr>
      <td>January 3, 2015</td><td>OLD</td><td>Old Co</td>
      <td>ANCIENT</td><td>Ancient Co</td><td>Long before our window.</td><td>[99]</td>
    </tr>
    <tr>
      <td>not a date</td><td>X</td><td>X Co</td><td></td><td></td><td>Bad row.</td>
    </tr>
  </tbody>
</table>
</body></html>
"""


class TestParseConstituentsTable:
    def test_extracts_tickers_in_order(self) -> None:
        assert parse_constituents_table(_CONSTITUENTS_HTML) == ["MMM", "AAPL", "BRK.B"]

    def test_missing_table_raises(self) -> None:
        with pytest.raises(ValueError, match="constituents"):
            parse_constituents_table("<html><body>nothing here</body></html>")


class TestParseChangesTable:
    def test_skips_both_header_rows(self) -> None:
        changes = parse_changes_table(_CHANGES_HTML)
        # 5 rows in the fixture, minus the unparseable-date row = 4.
        assert len(changes) == 4

    def test_parses_add_and_remove_row(self) -> None:
        changes = parse_changes_table(_CHANGES_HTML)
        first = changes[0]
        assert first == MembershipChange(
            effective_date=dt.date(2026, 8, 18), added="RDDT", removed="AVB", reason="Merger."
        )

    def test_removal_only_row_has_none_added(self) -> None:
        changes = parse_changes_table(_CHANGES_HTML)
        removal_only = next(c for c in changes if c.removed == "CAG")
        assert removal_only.added is None

    def test_unparseable_date_row_is_skipped_not_raised(self) -> None:
        changes = parse_changes_table(_CHANGES_HTML)
        assert all(c.added != "X" for c in changes)

    def test_missing_table_raises(self) -> None:
        with pytest.raises(ValueError, match="changes"):
            parse_changes_table("<html><body>nothing here</body></html>")


class TestSp500EverMembers:
    def _changes(self) -> list[MembershipChange]:
        return parse_changes_table(_CHANGES_HTML)

    def test_includes_current_constituents(self) -> None:
        result = sp500_ever_members(
            "2026-01-01", "2026-12-31", current=["AAPL", "MSFT"], changes=[]
        )
        assert result == ["AAPL", "MSFT"]

    def test_includes_added_and_removed_within_window(self) -> None:
        result = sp500_ever_members(
            "2026-01-01", "2026-12-31", current=["AAPL"], changes=self._changes()
        )
        # RDDT/AVB (Aug 2026) and CAG (June 2026) fall in-window; NOV/BRO (2021) and
        # OLD/ANCIENT (2015) do not.
        assert set(result) == {"AAPL", "RDDT", "AVB", "CAG"}

    def test_excludes_changes_outside_window(self) -> None:
        result = sp500_ever_members(
            "2015-06-01", "2015-06-02", current=["AAPL"], changes=self._changes()
        )
        assert set(result) == {"AAPL"}  # none of the fixture rows fall in this window

    def test_5_year_window_covers_a_removed_then_readded_style_history(self) -> None:
        # A stock removed near the start of a 5y window and never seen again in
        # `current` must still appear — this is the core survivorship-bias check.
        result = sp500_ever_members(
            "2021-09-01", "2021-10-01", current=["AAPL"], changes=self._changes()
        )
        assert "NOV" in result
        assert "BRO" in result

    def test_defaults_to_today_when_end_omitted(self) -> None:
        result = sp500_ever_members(dt.date.today().isoformat(), current=["AAPL"], changes=[])
        assert result == ["AAPL"]

    def test_deduplicates_and_sorts(self) -> None:
        result = sp500_ever_members(
            "2026-01-01",
            "2026-12-31",
            current=["AAPL", "RDDT"],
            changes=self._changes(),
        )
        assert result == sorted(set(result))
