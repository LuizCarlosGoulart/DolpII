"""Config-driven field resolver for canonical NFS-e fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml

from src.core import FieldCandidate, FieldResolver, ResolvedField, load_field_dictionary


@dataclass
class _ScoredCandidate:
    field_name: str
    candidate: FieldCandidate
    confidence: float
    evidence: dict[str, float | str | bool]


class ConfigDrivenFieldResolver(FieldResolver):
    """Resolve extracted candidates into canonical fields using config-driven evidence."""

    def __init__(
        self,
        *,
        config_dir: str | Path | None = None,
        resolver_name: str = "config-driven-field-resolver",
    ) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else Path(__file__).resolve().parents[2] / "configs"
        self.resolver_name = resolver_name
        self.field_dictionary = load_field_dictionary(self.config_dir / "field_dictionary.yaml")
        self.field_map = self.field_dictionary.by_internal_name()
        self.aliases = self._load_yaml("field_aliases.yaml").get("aliases", {})
        self.patterns = self._load_yaml("field_patterns.yaml").get("patterns", {})
        self.weights = self._load_yaml("scoring_rules.yaml")
        self.thresholds = self._load_yaml("decision_thresholds.yaml").get("thresholds", {})
        heuristics = self.weights.get("heuristics", {})
        self.section_keywords = heuristics.get("section_keywords", {})
        self.same_block_boost = float(heuristics.get("same_block_boost", 0.75))
        self.ambiguity_delta = float(self.thresholds.get("ambiguity_delta", 0.10))

    def resolve(
        self,
        document,
        candidates: list[FieldCandidate],
    ) -> list[ResolvedField]:
        scored_by_field: dict[str, list[_ScoredCandidate]] = {}
        for candidate in candidates:
            for scored in self._score_candidate(candidate):
                scored_by_field.setdefault(scored.field_name, []).append(scored)

        resolved_fields: list[ResolvedField] = []
        threshold = float(self.thresholds.get("minimum_field_confidence", 0.60))

        for field_name, scored_candidates in scored_by_field.items():
            ordered = sorted(scored_candidates, key=lambda item: item.confidence, reverse=True)
            top = ordered[0]
            second = ordered[1] if len(ordered) > 1 else None
            alternatives = [
                {
                    "candidate_id": item.candidate.candidate_id,
                    "value": item.candidate.value,
                    "confidence": item.confidence,
                    "evidence": item.evidence,
                }
                for item in ordered
            ]

            ambiguous = second is not None and abs(top.confidence - second.confidence) <= self.ambiguity_delta
            low_confidence = top.confidence < threshold

            resolved_fields.append(
                ResolvedField(
                    field_name=field_name,
                    value=None if ambiguous or low_confidence else top.candidate.value,
                    status="conflict" if ambiguous or low_confidence else "resolved",
                    confidence=top.confidence,
                    source_candidate_ids=[item.candidate.candidate_id for item in ordered],
                    resolver_name=self.resolver_name,
                    metadata={
                        "selected_candidate_id": top.candidate.candidate_id,
                        "selected_evidence": top.evidence,
                        "alternatives": alternatives,
                        "ambiguity_detected": ambiguous,
                        "low_confidence": low_confidence,
                        "document_id": getattr(document, "document_id", None),
                    },
                )
            )

        return sorted(resolved_fields, key=lambda field: field.field_name)

    def _score_candidate(self, candidate: FieldCandidate) -> list[_ScoredCandidate]:
        scored: list[_ScoredCandidate] = []
        for canonical_name, definition in self.field_map.items():
            evidence = self._collect_evidence(candidate, canonical_name, definition.category)
            score = self._combine_evidence(candidate, evidence)
            if score > 0:
                scored.append(
                    _ScoredCandidate(
                        field_name=canonical_name,
                        candidate=candidate,
                        confidence=score,
                        evidence=evidence,
                    )
                )
        return scored

    def _collect_evidence(
        self,
        candidate: FieldCandidate,
        canonical_name: str,
        category: str,
    ) -> dict[str, float | str | bool]:
        aliases = {canonical_name, *self.aliases.get(canonical_name, [])}
        aliases_normalized = {self._normalize_text(value) for value in aliases}

        field_name = self._normalize_text(candidate.field_name)
        label_text = self._normalize_text(candidate.metadata.get("label_text"))
        section_text = self._normalize_text(candidate.metadata.get("section_name") or candidate.metadata.get("section"))
        context_text = self._normalize_text(candidate.metadata.get("context_text"))

        alias_match = 1.0 if field_name in aliases_normalized or label_text in aliases_normalized else 0.0
        if alias_match == 0.0:
            alias_match = max(
                (0.7 for alias in aliases_normalized if alias and alias in {field_name, label_text, context_text}),
                default=0.0,
            )

        pattern_match = 0.0
        for pattern in self.patterns.get(canonical_name, []):
            if re.search(pattern, candidate.value):
                pattern_match = 1.0
                break

        section_keywords = self.section_keywords.get(category, ())
        section_match = max(
            (1.0 for keyword in section_keywords if keyword and keyword in section_text),
            default=0.0,
        )
        if section_match == 0.0:
            section_match = max(
                (0.5 for keyword in section_keywords if keyword and keyword in context_text),
                default=0.0,
            )

        label_distance = candidate.metadata.get("label_distance")
        distance_score = 0.0
        if isinstance(label_distance, (int, float)):
            distance_score = max(0.0, 1.0 - min(float(label_distance) / 200.0, 1.0))

        same_block = candidate.metadata.get("same_block_as_label") is True
        if not same_block:
            same_block = candidate.metadata.get("block_num") is not None and candidate.metadata.get("block_num") == candidate.metadata.get("label_block_num")

        positional_match = max(distance_score, self.same_block_boost if same_block else 0.0)

        section_conflict = self._detect_section_conflict(category, section_text)

        return {
            "alias_match": alias_match,
            "pattern_match": pattern_match,
            "section_match": section_match,
            "positional_match": positional_match,
            "section_conflict": section_conflict,
            "section_text": section_text,
        }

    def _combine_evidence(
        self,
        candidate: FieldCandidate,
        evidence: dict[str, float | str | bool],
    ) -> float:
        evidence_weights = self.weights.get("evidence_weights", {})
        confidence_weights = self.weights.get("confidence_weights", {})
        penalties = self.weights.get("penalties", {})

        alias_score = float(evidence["alias_match"]) * float(evidence_weights.get("alias_match", 0.0))
        pattern_score = float(evidence["pattern_match"]) * float(evidence_weights.get("regex_pattern_match", 0.0))
        positional_score = float(evidence["positional_match"]) * float(evidence_weights.get("positional_context_match", 0.0))
        section_score = float(evidence["section_match"]) * float(evidence_weights.get("section_context_match", 0.0))
        engine_score = float(candidate.confidence or 0.0) * float(confidence_weights.get("engine_confidence", 0.0))
        penalty = float(penalties.get("section_conflict", 0.0)) if evidence["section_conflict"] else 0.0

        total = alias_score + pattern_score + positional_score + section_score + engine_score - penalty
        max_score = (
            float(evidence_weights.get("alias_match", 0.0))
            + float(evidence_weights.get("regex_pattern_match", 0.0))
            + float(evidence_weights.get("positional_context_match", 0.0))
            + float(evidence_weights.get("section_context_match", 0.0))
            + float(confidence_weights.get("engine_confidence", 0.0))
        )
        if max_score <= 0:
            return 0.0
        return max(0.0, min(total / max_score, 1.0))

    @staticmethod
    def _detect_section_conflict(category: str, section_text: str) -> bool:
        if not section_text:
            return False
        if category == "provider" and "tomador" in section_text:
            return True
        if category == "recipient" and "prestador" in section_text:
            return True
        return False

    @staticmethod
    def _normalize_text(value: object) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def _load_yaml(self, filename: str) -> dict:
        with (self.config_dir / filename).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
