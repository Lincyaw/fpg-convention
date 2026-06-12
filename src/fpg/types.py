"""Base types: entity reference, time interval, evidence."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .vocab import QueryLanguage

# Entity reference: an identifier in the topology-layer entity catalog,
# formatted "<prefix>:<name>". The structural pattern below accepts any
# well-formed prefix; whether the prefix is *registered* (core types plus
# testbed extensions, see fpg.entities) is a separate strict-layer check so
# that schema validation never depends on installed extensions. The name part
# allows letters, digits, ".", "_", "-", ">" (reified relational entities
# such as link:checkout->order).
EntityRef = Annotated[
    str,
    Field(
        pattern=r"^[a-z][a-z0-9_]*:[A-Za-z0-9][A-Za-z0-9._>-]*$",
        description=(
            "Topology-layer entity reference, formatted '<prefix>:<name>'. "
            "The prefix must be an entity type declared in the system's "
            "vocabulary profile (e.g. svc/pod/job/stage); registration is "
            "checked by tooling, not by this pattern. Reified relational "
            "entity example: link:checkout->order"
        ),
        examples=[
            "svc:checkout",
            "pod:order-7d9f-2",
            "link:checkout->order",
            "cfg:svc-order.db_pool_size",
            "slo:checkout-latency",
        ],
    ),
]


class TimeInterval(BaseModel):
    """Time interval [start, end]. Instantaneous events use a degenerate
    interval (start == end).
    """

    model_config = ConfigDict(frozen=True)

    start: datetime = Field(description="Interval start (ISO 8601 with timezone)")
    end: datetime = Field(
        description="Interval end (ISO 8601 with timezone); "
        "instantaneous events have end == start"
    )

    @model_validator(mode="after")
    def _start_not_after_end(self) -> "TimeInterval":
        if self.start > self.end:
            raise ValueError(
                f"time interval start ({self.start}) is after end ({self.end})"
            )
        return self

    @property
    def is_instantaneous(self) -> bool:
        return self.start == self.end

    def iou(self, other: "TimeInterval") -> float:
        """IoU with another interval, used for evaluation-side time
        alignment. Two coincident degenerate intervals score 1.
        """
        latest_start = max(self.start, other.start)
        earliest_end = min(self.end, other.end)
        if latest_start > earliest_end:
            return 0.0
        inter = (earliest_end - latest_start).total_seconds()
        union = (
            max(self.end, other.end) - min(self.start, other.start)
        ).total_seconds()
        if union == 0:
            return 1.0
        return inter / union


class EvidenceQuery(BaseModel):
    language: QueryLanguage = Field(
        description="Query language of the statement, one of "
        "vocab.QueryLanguage {sql, promql, logql, http, other}"
    )
    statement: str = Field(
        min_length=1,
        description="The executable query text, e.g. a PromQL expression, "
        "an SQL statement, or an HTTP request line",
    )


class Evidence(BaseModel):
    query: EvidenceQuery = Field(
        description="Re-executable query reproducing the observation"
    )
    explanation: str = Field(
        min_length=1,
        description="explanation of the evidence",
    )
