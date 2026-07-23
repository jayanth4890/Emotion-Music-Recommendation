"""
Music Recommendation Engine Module.
Generates properly encoded YouTube and Spotify music search URLs based on
detected emotion, user language, target artist, and genre preferences.
"""

import urllib.parse
from typing import Dict, List


class MusicRecommenderEngine:
    """
    Recommendation engine mapping emotional states and user preferences to music search queries.
    """

    MOOD_DESCRIPTIONS: Dict[str, str] = {
        "Happy": "Upbeat, joyful, and energetic tracks to elevate your mood!",
        "Sad": "Melancholic, soothing, and acoustic melodies for comfort.",
        "Angry": "High-energy rock, metal, or intense beats to release tension.",
        "Fear": "Calming, ambient, and peaceful soundscapes to relax your mind.",
        "Surprise": "Dynamic, vibrant, and unexpected tunes to match your surprise!",
        "Neutral": "Chill lo-fi, background instrumental, and easy-listening tracks.",
        "Disgust": "Experimental, dark synth, or rhythm-heavy music to reset your vibe."
    }

    def __init__(self):
        pass

    def get_mood_description(self, emotion: str) -> str:
        """
        Returns a human-readable mood description for the given emotion.

        Args:
            emotion (str): Detected emotion name.

        Returns:
            str: Mood description.
        """
        if not emotion:
            return "Start the camera to detect your emotional state."
        return self.MOOD_DESCRIPTIONS.get(
            emotion.capitalize(),
            "Personalized music recommendations tailored to your mood."
        )

    def generate_youtube_url(
        self,
        emotion: str,
        language: str = "",
        artist: str = "",
        genre: str = ""
    ) -> str:
        """
        Generates a URL-encoded YouTube search URL.

        Args:
            emotion (str): Detected emotion.
            language (str): Preferred language.
            artist (str): Target artist/singer.
            genre (str): Preferred genre.

        Returns:
            str: Encoded YouTube search URL.
        """
        query_parts: List[str] = []
        if language:
            query_parts.append(language)
        if genre and genre.lower() != "any":
            query_parts.append(genre)

        mood_tag = f"{emotion} songs" if emotion else "music"
        query_parts.append(mood_tag)

        if artist:
            query_parts.append(artist)

        full_query = " ".join(query_parts).strip()
        encoded_query = urllib.parse.quote_plus(full_query)
        return f"https://www.youtube.com/results?search_query={encoded_query}"

    def generate_spotify_url(
        self,
        emotion: str,
        language: str = "",
        artist: str = "",
        genre: str = ""
    ) -> str:
        """
        Generates a URL-encoded Spotify search URL.

        Args:
            emotion (str): Detected emotion.
            language (str): Preferred language.
            artist (str): Target artist/singer.
            genre (str): Preferred genre.

        Returns:
            str: Encoded Spotify search URL.
        """
        query_parts: List[str] = []
        if language:
            query_parts.append(language)
        if genre and genre.lower() != "any":
            query_parts.append(genre)

        if emotion:
            query_parts.append(emotion)
        query_parts.append("songs")

        if artist:
            query_parts.append(artist)

        full_query = " ".join(query_parts).strip()
        encoded_query = urllib.parse.quote(full_query)
        return f"https://open.spotify.com/search/{encoded_query}"
