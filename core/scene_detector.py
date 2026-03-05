"""
core/scene_detector.py
Intelligent scene boundary detection using:
  1. Semantic drift (cosine distance between rolling windows)
  2. Temporal gaps between subtitle cues
  3. Content transition signals (question marks, speaker changes, topic shifts)
  4. Minimum/maximum scene duration constraints
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from core.subtitle_parser import CueEntry
from core.embeddings import (
    TFIDFVectorizer, cosine_similarity,
    analyse_sentiment, extract_brand_safety,
    classify_iab, extract_entities, extract_topics,
)


@dataclass
class Scene:
    scene_id: str
    video_id: str
    index: int
    start_ms: int
    end_ms: int
    cues: list[CueEntry]
    text: str
    # Filled by analysis
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    sentiment: dict = field(default_factory=dict)
    brand_safety: dict = field(default_factory=dict)
    iab_categories: list[dict] = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    engagement_score: float = 0.0
    ad_suitability: float = 0.0
    is_boundary: bool = True

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_sec(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def fmt_time(self, ms: int) -> str:
        h = ms // 3_600_000
        m = (ms % 3_600_000) // 60_000
        s = (ms % 60_000) // 1000
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def start_fmt(self) -> str:
        return self.fmt_time(self.start_ms)

    @property
    def end_fmt(self) -> str:
        return self.fmt_time(self.end_ms)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "video_id": self.video_id,
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration_sec": self.duration_sec,
            "start_fmt": self.start_fmt,
            "end_fmt": self.end_fmt,
            "text": self.text,
            "word_count": len(self.text.split()),
            "sentiment": self.sentiment,
            "brand_safety": self.brand_safety,
            "iab_categories": self.iab_categories,
            "entities": self.entities,
            "topics": self.topics,
            "engagement_score": self.engagement_score,
            "ad_suitability": self.ad_suitability,
        }


class SceneDetector:
    """
    Detects natural scene boundaries using a multi-signal approach:

    Signal 1 — Semantic drift: cosine distance between overlapping windows
               of cues. High drift = topic/scene change.
    Signal 2 — Temporal gap: long pauses between cues suggest scene breaks.
    Signal 3 — Content signals: "?", speaker tags, "Previously on...", etc.
    Signal 4 — Duration constraints: scenes must be between min/max seconds.
    """

    def __init__(
        self,
        min_scene_sec: float = 20.0,
        max_scene_sec: float = 120.0,
        semantic_threshold: float = 0.35,
        temporal_gap_sec: float = 3.0,
        window_size: int = 4,
    ):
        self.min_scene_sec = min_scene_sec
        self.max_scene_sec = max_scene_sec
        self.semantic_threshold = semantic_threshold
        self.temporal_gap_sec = temporal_gap_sec
        self.window_size = window_size
        self.vectorizer: Optional[TFIDFVectorizer] = None

    # ── Boundary signals ─────────────────────────────────────────────────────
    def _has_content_transition(self, cue: CueEntry, prev_cue: Optional[CueEntry]) -> bool:
        """Detect hard transition signals in cue text."""
        t = cue.text.lower()
        transition_patterns = [
            r"previously on", r"next time", r"coming up", r"meanwhile",
            r"later that", r"the next day", r"chapter \d", r"part \d",
            r"act \w+", r"\[scene\]", r"\[cut to\]", r"\[fade",
        ]
        for p in transition_patterns:
            if re.search(p, t):
                return True
        # Speaker change: "NAME:" format
        if re.match(r"^[A-Z][A-Z\s]{2,15}:\s", cue.text):
            if prev_cue and not re.match(r"^[A-Z][A-Z\s]{2,15}:\s", prev_cue.text):
                return True
        return False

    def _temporal_gap(self, a: CueEntry, b: CueEntry) -> float:
        """Gap in seconds between end of cue a and start of cue b."""
        return (b.start_ms - a.end_ms) / 1000.0

    def _window_text(self, cues: list[CueEntry], center: int, half: int = 2) -> str:
        start = max(0, center - half)
        end = min(len(cues), center + half + 1)
        return " ".join(c.text for c in cues[start:end])

    # ── Main detection ───────────────────────────────────────────────────────
    def detect(self, cues: list[CueEntry], video_id: str) -> list[Scene]:
        if not cues:
            return []

        # Build vocabulary from all cues
        all_texts = [c.text for c in cues]
        self.vectorizer = TFIDFVectorizer(max_features=1500)
        self.vectorizer.fit(all_texts)

        # Compute windowed embeddings for semantic drift
        window_texts = [self._window_text(cues, i, self.window_size // 2)
                        for i in range(len(cues))]
        embeddings = self.vectorizer.transform(window_texts)

        # Score each potential boundary
        boundary_scores = np.zeros(len(cues))
        for i in range(1, len(cues)):
            # Semantic drift
            drift = 1.0 - cosine_similarity(embeddings[i], embeddings[i - 1])
            boundary_scores[i] += drift * 0.6

            # Temporal gap
            gap = self._temporal_gap(cues[i - 1], cues[i])
            if gap > self.temporal_gap_sec:
                boundary_scores[i] += min(gap / 10.0, 0.4) * 0.25

            # Content transition
            if self._has_content_transition(cues[i], cues[i - 1]):
                boundary_scores[i] += 0.15

        # Build scenes using adaptive threshold
        threshold = max(self.semantic_threshold,
                        float(np.percentile(boundary_scores[1:], 65)) if len(boundary_scores) > 2 else self.semantic_threshold)

        scenes: list[Scene] = []
        scene_start_idx = 0

        def _flush(start_idx: int, end_idx: int, scene_index: int) -> Scene:
            scene_cues = cues[start_idx:end_idx + 1]
            text = " ".join(c.text for c in scene_cues)
            return Scene(
                scene_id=f"{video_id}_s{scene_index:03d}",
                video_id=video_id,
                index=scene_index,
                start_ms=scene_cues[0].start_ms,
                end_ms=scene_cues[-1].end_ms,
                cues=scene_cues,
                text=text,
            )

        for i in range(1, len(cues)):
            current_start_ms = cues[scene_start_idx].start_ms
            current_duration = (cues[i].end_ms - current_start_ms) / 1000.0

            is_boundary = (
                boundary_scores[i] >= threshold
                and current_duration >= self.min_scene_sec
            )
            force_split = current_duration >= self.max_scene_sec

            if is_boundary or force_split:
                scene = _flush(scene_start_idx, i - 1, len(scenes))
                scenes.append(scene)
                scene_start_idx = i

        # Final scene
        if scene_start_idx < len(cues):
            final = _flush(scene_start_idx, len(cues) - 1, len(scenes))
            # Merge very short trailing scene into previous
            if scenes and final.duration_sec < self.min_scene_sec / 2:
                prev = scenes[-1]
                merged_cues = prev.cues + final.cues
                merged_text = " ".join(c.text for c in merged_cues)
                scenes[-1] = Scene(
                    scene_id=prev.scene_id,
                    video_id=video_id,
                    index=prev.index,
                    start_ms=prev.start_ms,
                    end_ms=final.end_ms,
                    cues=merged_cues,
                    text=merged_text,
                )
            else:
                scenes.append(final)

        # Analyse each scene
        self._analyse_scenes(scenes)
        return scenes

    def _analyse_scenes(self, scenes: list[Scene]) -> None:
        """Fill metadata fields for each scene in-place."""
        if not scenes or self.vectorizer is None:
            return

        # Batch embed all scene texts
        all_texts = [s.text for s in scenes]
        embeddings = self.vectorizer.transform(all_texts)

        for i, scene in enumerate(scenes):
            scene.embedding = embeddings[i]
            scene.sentiment = analyse_sentiment(scene.text)
            scene.brand_safety = extract_brand_safety(scene.text)
            scene.iab_categories = classify_iab(scene.text)
            scene.entities = extract_entities(scene.text)
            scene.topics = extract_topics(scene.text)
            scene.engagement_score = self._score_engagement(scene)
            scene.ad_suitability = self._score_ad_suitability(scene)

    def _score_engagement(self, scene: Scene) -> float:
        """
        Engagement score 0-1 based on:
        - Emotional intensity (sentiment)
        - Content density (word count relative to duration)
        - Presence of key moments (questions, exclamations)
        - Entity richness
        """
        intensity = scene.sentiment.get("intensity", 0.0)
        word_count = len(scene.text.split())
        density = min(word_count / max(scene.duration_sec, 1.0) / 3.0, 1.0)
        punctuation_score = (
            scene.text.count("!") * 0.1 + scene.text.count("?") * 0.05
        ) / max(word_count / 50, 1)
        entity_score = min(
            len(scene.entities.get("products", []))
            + len(scene.entities.get("organisations", [])),
            5,
        ) / 5.0
        score = (
            intensity * 0.35
            + density * 0.30
            + min(punctuation_score, 0.2)
            + entity_score * 0.15
        )
        return round(min(score, 1.0), 3)

    def _score_ad_suitability(self, scene: Scene) -> float:
        """
        Ad suitability 0-1:
        High score = great ad placement candidate.
        Penalised for unsafe content, very high emotional intensity (drama),
        boosted for positive sentiment + product mentions.
        """
        safety = scene.brand_safety.get("safety_score", 1.0)
        sentiment_score = scene.sentiment.get("score", 0.0)
        intensity = scene.sentiment.get("intensity", 0.0)

        positivity_bonus = max(0.0, sentiment_score * 0.2)
        intensity_penalty = max(0.0, (intensity - 0.5) * 0.3)
        product_bonus = min(len(scene.entities.get("products", [])) * 0.05, 0.2)

        score = (
            safety * 0.6
            + positivity_bonus
            + product_bonus
            - intensity_penalty
        )
        return round(max(0.0, min(score, 1.0)), 3)
