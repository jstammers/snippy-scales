"""S&P 500 historical membership, sourced from Wikipedia.

Used to build a survivorship-bias-free universe for a backfill — "every
ticker that was in the index at any point during a date range" — rather than
just today's 500 names, which would silently exclude every stock removed
since (e.g. SIVB, FRC, ATVI, TWTR).

Two Wikipedia pages are combined:

* `List of S&P 500 companies <https://en.wikipedia.org/wiki/List_of_S%26P_500_companies>`_
  — today's constituent list (``table#constituents``).
* `Historical components of the S&P 500 <https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500>`_
  — every addition/removal since 1976 (``table#changes``), split out of the
  first page in August 2026. If Wikipedia moves it again, only
  :data:`_CHANGES_URL` needs updating — the parsing logic keys off the
  ``id="changes"`` table, not page structure.

:func:`sp500_ever_members` doesn't need to reconstruct membership for every
day in the range — the union of (today's constituents) ∪ (every ticker
added or removed within the window) is exactly the set of tickers that were
members at some point in ``[start, end]``: a ticker removed inside the
window was a member immediately before its removal; a ticker added inside
the window is a member from that point on (already covered by "today's
constituents" unless later removed, which is itself an in-window removal
event covered by the same union).

Known limitation: a pure ticker-symbol rename with no index membership
change (the company neither leaves nor re-joins) won't appear as a row in
the changes table, so the old symbol won't be included even though it was
once the same constituent. Consult ``rename_overrides`` in the caller
(e.g. the backfill script) for known cases (e.g. FB -> META).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import requests
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from bs4.element import Tag

logger = logging.getLogger(__name__)

_CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
_USER_AGENT = "snippy-scales-data-ingestion/0.1 (+https://github.com/jstammers/snippy-scales)"
_REQUEST_TIMEOUT_S = 30.0

_MONTHS = {
    name: i
    for i, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
_DATE_RE = re.compile(r"^([A-Za-z]+) (\d{1,2}), (\d{4})$")


@dataclass(frozen=True)
class MembershipChange:
    """One row of the "Historical components" changes table.

    Attributes:
        effective_date: Date the change took effect.
        added: Ticker added to the index on this date, or ``None`` if this
            row only records a removal.
        removed: Ticker removed from the index on this date, or ``None`` if
            this row only records an addition.
        reason: Free-text reason (e.g. "Market capitalization changes.").
    """

    effective_date: dt.date
    added: str | None
    removed: str | None
    reason: str


def _fetch_html(url: str) -> str:
    """GET *url* with a descriptive User-Agent and raise on any HTTP error."""
    response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.text


def _data_rows(table: Tag) -> list[Tag]:
    """Return every ``<tr>`` in *table* that has at least one ``<td>`` — i.e.
    skips header row(s) regardless of whether the table has one or two of
    them, or whether the parser inserted an explicit ``<tbody>``.
    """
    return [row for row in table.find_all("tr") if row.find_all("td")]


def parse_constituents_table(html: str) -> list[str]:
    """Extract the ticker column from the ``table#constituents`` Wikipedia table.

    Args:
        html: Full page HTML of "List of S&P 500 companies".

    Returns:
        Tickers in table order (first column is "Symbol").

    Raises:
        ValueError: If no ``table#constituents`` is found.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="constituents")
    if table is None:
        raise ValueError(
            "No table#constituents found — Wikipedia's page structure may have changed."
        )
    return [row.find_all("td")[0].get_text(strip=True) for row in _data_rows(table)]


def _parse_effective_date(text: str) -> dt.date | None:
    """Parse a "Month D, YYYY" date string; return ``None`` if unparseable."""
    match = _DATE_RE.match(text.strip())
    if match is None:
        return None
    month_name, day_str, year_str = match.groups()
    month = _MONTHS.get(month_name)
    if month is None:
        return None
    return dt.date(int(year_str), month, int(day_str))


def parse_changes_table(html: str) -> list[MembershipChange]:
    """Extract every row of the ``table#changes`` Wikipedia table.

    Args:
        html: Full page HTML of "Historical components of the S&P 500".

    Returns:
        One :class:`MembershipChange` per data row, in the table's original
        (newest-first) order. Rows whose date cell doesn't parse are skipped
        with a warning rather than aborting the whole fetch.

    Raises:
        ValueError: If no ``table#changes`` is found.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="changes")
    if table is None:
        raise ValueError("No table#changes found — Wikipedia's page structure may have changed.")

    changes: list[MembershipChange] = []
    for row in _data_rows(table):
        cells = [c.get_text(strip=True) for c in row.find_all("td")]
        # Columns: Effective Date, Added Ticker, Added Security, Removed
        # Ticker, Removed Security, Reason, [Refs — not always present].
        if len(cells) < 6:
            logger.warning("Skipping changes-table row with unexpected shape: %r", cells)
            continue
        effective_date = _parse_effective_date(cells[0])
        if effective_date is None:
            logger.warning("Skipping changes-table row with unparseable date: %r", cells[0])
            continue
        changes.append(
            MembershipChange(
                effective_date=effective_date,
                added=cells[1] or None,
                removed=cells[3] or None,
                reason=cells[5],
            )
        )
    return changes


def fetch_current_constituents() -> list[str]:
    """Fetch today's S&P 500 constituent tickers from Wikipedia.

    Returns:
        Tickers in table order.
    """
    return parse_constituents_table(_fetch_html(_CONSTITUENTS_URL))


def fetch_membership_changes() -> list[MembershipChange]:
    """Fetch the full S&P 500 addition/removal history from Wikipedia (since 1976).

    Returns:
        Every :class:`MembershipChange`, newest first.
    """
    return parse_changes_table(_fetch_html(_CHANGES_URL))


def _to_date(value: str | dt.date) -> dt.date:
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(value)


def sp500_ever_members(
    start: str | dt.date,
    end: str | dt.date | None = None,
    *,
    current: list[str] | None = None,
    changes: list[MembershipChange] | None = None,
) -> list[str]:
    """Union of every ticker that was an S&P 500 member at any point in ``[start, end]``.

    See the module docstring for why this doesn't need day-by-day membership
    reconstruction.

    Args:
        start: Inclusive start date.
        end: Inclusive end date. Defaults to today.
        current: Today's constituents — fetched from Wikipedia if omitted.
            Pass explicitly to avoid a network call (e.g. in tests, or to
            reuse a result already fetched for logging/caching).
        changes: Full addition/removal history — fetched from Wikipedia if
            omitted.

    Returns:
        Sorted, deduplicated list of tickers.
    """
    start_date = _to_date(start)
    end_date = _to_date(end) if end is not None else dt.date.today()

    current_members = current if current is not None else fetch_current_constituents()
    change_history = changes if changes is not None else fetch_membership_changes()

    members = set(current_members)
    for change in change_history:
        if start_date <= change.effective_date <= end_date:
            if change.added:
                members.add(change.added)
            if change.removed:
                members.add(change.removed)

    return sorted(members)
