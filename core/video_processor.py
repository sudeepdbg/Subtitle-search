"""
core/video_processor.py
Orchestrates the full pipeline for a single video:
  parse subtitles → detect scenes → analyse → build video-level metadata
"""
from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from core.subtitle_parser import CueEntry, parse_subtitle_file, parse_youtube_transcript
from core.scene_detector import Scene, SceneDetector
from core.embeddings import (
    TFIDFVectorizer, classify_iab, extract_entities, extract_topics,
    analyse_sentiment, _IAB_NAMES,
)


@dataclass
class VideoMetadata:
    video_id: str
    title: str
    source: str            # "upload" | "youtube"
    duration_ms: int
    total_cues: int
    scenes: list[Scene]
    # Video-level analysis
    dominant_iab: list[dict] = field(default_factory=list)
    overall_sentiment: dict = field(default_factory=dict)
    emotional_arc: list[dict] = field(default_factory=list)
    key_scenes: list[str] = field(default_factory=list)  # scene_ids
    narrative_structure: str = ""
    franchise_themes: list[str] = field(default_factory=list)
    youtube_meta: Optional[dict] = None

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    def fmt_duration(self) -> str:
        ms = self.duration_ms
        h = ms // 3_600_000
        m = (ms % 3_600_000) // 60_000
        s = (ms % 60_000) // 1000
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m}m {s:02d}s"

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "source": self.source,
            "duration_ms": self.duration_ms,
            "duration_fmt": self.fmt_duration(),
            "total_cues": self.total_cues,
            "scene_count": self.scene_count,
            "dominant_iab": self.dominant_iab,
            "overall_sentiment": self.overall_sentiment,
            "narrative_structure": self.narrative_structure,
            "franchise_themes": self.franchise_themes,
            "key_scenes": self.key_scenes,
            "youtube_meta": self.youtube_meta,
        }


class VideoProcessor:

    def __init__(
        self,
        min_scene_sec: float = 20.0,
        max_scene_sec: float = 120.0,
        semantic_threshold: float = 0.35,
    ):
        self.detector = SceneDetector(
            min_scene_sec=min_scene_sec,
            max_scene_sec=max_scene_sec,
            semantic_threshold=semantic_threshold,
        )

    def _make_video_id(self, title: str) -> str:
        return "v_" + hashlib.md5(title.encode()).hexdigest()[:10]

    def process_file(
        self,
        content: str,
        filename: str,
        fmt: Optional[str] = None,
    ) -> VideoMetadata:
        """Process an uploaded SRT or VTT file."""
        title = re.sub(r"\.(srt|vtt)$", "", filename, flags=re.IGNORECASE)
        video_id = self._make_video_id(filename)
        cues = parse_subtitle_file(content, fmt)
        return self._build_video(cues, video_id, title, "upload")

    def process_youtube_transcript(
        self,
        transcript_data: list[dict],
        video_id: str,
        title: str,
        youtube_meta: Optional[dict] = None,
    ) -> VideoMetadata:
        """Process YouTube transcript API data."""
        cues = parse_youtube_transcript(transcript_data)
        vm = self._build_video(cues, f"yt_{video_id}", title, "youtube")
        vm.youtube_meta = youtube_meta
        return vm

    def _build_video(
        self,
        cues: list[CueEntry],
        video_id: str,
        title: str,
        source: str,
    ) -> VideoMetadata:
        if not cues:
            return VideoMetadata(
                video_id=video_id, title=title, source=source,
                duration_ms=0, total_cues=0, scenes=[],
            )

        duration_ms = cues[-1].end_ms
        scenes = self.detector.detect(cues, video_id)

        vm = VideoMetadata(
            video_id=video_id,
            title=title,
            source=source,
            duration_ms=duration_ms,
            total_cues=len(cues),
            scenes=scenes,
        )

        self._analyse_video(vm)
        return vm

    def _analyse_video(self, vm: VideoMetadata) -> None:
        if not vm.scenes:
            return

        full_text = " ".join(s.text for s in vm.scenes)

        # Dominant IAB from all text
        vm.dominant_iab = classify_iab(full_text, top_n=5)

        # Overall sentiment
        vm.overall_sentiment = analyse_sentiment(full_text)

        # Emotional arc (sentiment per scene)
        vm.emotional_arc = [
            {
                "scene_index": s.index,
                "start_sec": s.start_sec,
                "sentiment_score": s.sentiment.get("score", 0.0),
                "sentiment_label": s.sentiment.get("label", "neutral"),
                "intensity": s.sentiment.get("intensity", 0.0),
                "engagement": s.engagement_score,
            }
            for s in vm.scenes
        ]

        # Key scenes = top 3 by engagement
        sorted_by_eng = sorted(vm.scenes, key=lambda s: s.engagement_score, reverse=True)
        vm.key_scenes = [s.scene_id for s in sorted_by_eng[:3]]

        # Narrative structure heuristic
        vm.narrative_structure = self._classify_narrative(vm)

        # Franchise themes
        vm.franchise_themes = extract_topics(full_text)[:6]

    def _classify_narrative(self, vm: VideoMetadata) -> str:
        """Classify narrative arc based on emotional trajectory."""
        if len(vm.emotional_arc) < 3:
            return "Short Form"
        scores = [e["sentiment_score"] for e in vm.emotional_arc]
        n = len(scores)
        first_third = sum(scores[:n//3]) / max(n//3, 1)
        last_third  = sum(scores[2*n//3:]) / max(n - 2*n//3, 1)
        mid_min     = min(scores[n//3:2*n//3]) if n > 3 else 0

        if last_third > first_third + 0.2:
            return "Rising Arc (positive resolution)"
        if last_third < first_third - 0.2:
            return "Falling Arc (dramatic tension)"
        if mid_min < -0.3 and last_third > -0.1:
            return "Classic Arc (conflict → resolution)"
        if abs(last_third - first_third) < 0.1:
            return "Flat / Documentary"
        return "Complex / Non-linear"


# ── YouTube API helper ────────────────────────────────────────────────────────
def fetch_youtube_metadata(video_id: str, api_key: str) -> Optional[dict]:
    """
    Fetch video metadata from YouTube Data API v3.
    Returns None if unavailable (no key, network error, etc.)
    """
    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=api_key)
        resp = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id,
        ).execute()
        items = resp.get("items", [])
        if not items:
            return None
        item = items[0]
        snippet = item.get("snippet", {})
        stats   = item.get("statistics", {})
        return {
            "id": video_id,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", "")[:500],
            "channel": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
            "tags": snippet.get("tags", [])[:20],
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
            "duration": item.get("contentDetails", {}).get("duration", ""),
        }
    except Exception:
        return None


def fetch_youtube_transcript(video_id: str) -> Optional[list[dict]]:
    """Fetch transcript via youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception:
        return None


def fetch_youtube_comments(video_id: str, api_key: str, max_results: int = 100) -> list[dict]:
    """Fetch top comments with sentiment for engagement analysis."""
    try:
        from googleapiclient.discovery import build
        from core.embeddings import analyse_sentiment
        youtube = build("youtube", "v3", developerKey=api_key)
        resp = youtube.commentThreads().list(
            part="snippet", videoId=video_id,
            maxResults=max_results, order="relevance",
        ).execute()
        comments = []
        for item in resp.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            sentiment = analyse_sentiment(text)
            comments.append({
                "text": text[:200],
                "likes": item["snippet"]["topLevelComment"]["snippet"].get("likeCount", 0),
                "sentiment": sentiment,
            })
        return comments
    except Exception:
        return []
