"""Loading a measurement's stored analysis as a fit's data input.

An analysis is a table of features extracted from one measurement — degradation
modes per RPT, ECM parameters per EIS sweep — stored alongside the measurement
it came from. It is the counterpart to :meth:`DataLoader.from_db` for feature
tables rather than traces, and deliberately not a :class:`DataLoader` subclass:
a ``DataLoader`` validates that its table is a voltage trace and derives steps,
cycles and experiments from it, none of which an analysis has.
"""

from __future__ import annotations

from typing import Any


class AnalysisLoader:
    """One of a measurement's stored analyses, fetched on demand.

    Every selector field given must match, so two analyses sharing a type are
    separated by name. Nothing prevents a measurement from carrying several
    analyses of one type — the database constrains only auto-generated ones — so
    a selector matching more than one raises rather than picking whichever was
    created last.

    Parameters
    ----------
    measurement_id : str
        Measurement the analysis is stored against.
    analysis_id : str, optional
        The analysis' own id. Identifies the row on its own, so any other
        fields given are ignored; this is what a resolved platform run records.
    analysis_type : str, optional
        Kind of analysis, e.g. ``"lam_lli_from_rpt"``. Free-form on the
        platform, so any string is accepted.
    name : str, optional
        The analysis' name, as given when it was created.
    client : ionworks.Ionworks, optional
        Pre-configured client. A default one is created from the environment
        when the data is first read.

    Raises
    ------
    ValueError
        If ``measurement_id`` is empty, or no selector field is given — an
        empty selector matches every analysis on the measurement.

    Examples
    --------
    >>> loader = AnalysisLoader("9f3c...", analysis_type="lam_lli_from_rpt")
    >>> loader.to_config()
    {'data': 'db:9f3c...', 'analysis': {'analysis_type': 'lam_lli_from_rpt'}}
    """

    def __init__(
        self,
        measurement_id: str,
        *,
        analysis_id: str | None = None,
        analysis_type: str | None = None,
        name: str | None = None,
        client: Any = None,
    ) -> None:
        if not measurement_id:
            raise ValueError(
                "AnalysisLoader needs a measurement_id: an analysis is stored "
                "against a measurement, so there is nothing to search without one."
            )
        selector = {
            "analysis_id": analysis_id,
            "analysis_type": analysis_type,
            "name": name,
        }
        # str(): a StrEnum like AnalysisType compares equal to its value without
        # being it, so the enum would reach to_config as a different type.
        self._selector = {
            key: str(value) for key, value in selector.items() if value is not None
        }
        if not self._selector:
            raise ValueError(
                "AnalysisLoader needs at least one of analysis_id, "
                "analysis_type or name; an empty selector matches every "
                "analysis on the measurement."
            )
        self._measurement_id = measurement_id
        self._client = client
        self._data = None
        self._resolved_id: str | None = self._selector.get("analysis_id")

    @property
    def analysis_id(self) -> str | None:
        """The analysis' id, once known.

        Set from the start when the selector named an id, and otherwise only
        after :attr:`data` has resolved the selector.
        """
        return self._resolved_id

    @property
    def data(self):
        """The analysis' feature table, fetched on first read and kept.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            Whichever the active client backend returns.
        """
        if self._data is None:
            client = self._client
            if client is None:
                from ionworks import Ionworks

                client = Ionworks()
                self._client = client
            found = client.analysis.find_one(self._measurement_id, **self._selector)
            self._resolved_id = found.id
            self._data = client.analysis.get_data(found.id)
        return self._data

    def to_config(self) -> dict[str, Any]:
        """Convert to the data-payload configuration a fit config carries.

        Returns
        -------
        dict
            ``{"data": "db:<measurement_id>", "analysis": {<selector>}}``.
        """
        return {
            "data": f"db:{self._measurement_id}",
            "analysis": dict(self._selector),
        }

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self._selector.items())
        return f"AnalysisLoader({self._measurement_id!r}, {fields})"
