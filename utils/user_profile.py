import re
from typing import Any, Dict, List


def split_by_punctuation(text: str) -> List[str]:
    """
    split logic: only use English punctuation (comma, semicolon, period, colon, dash, etc.) as separator,
    keep phrases inside hyphen `-`, e.g. self-insert.
    """
    tokens = re.split(r"[.,;:!?()\[\]{}—]+", text)  # note: do not include normal `-`
    tokens = [t.strip() for t in tokens if t.strip()]
    return tokens


class UserProfile:
    """Enhanced user profile with partial tag selection for testing incomplete information scenarios"""

    def __init__(self, username: str, full_profile: str):
        self.username = username
        self.full_profile = full_profile  # full user preference description

        # use new split function to extract all tags
        self.all_tags = split_by_punctuation(full_profile)

        # number of partial tags for testing (simulate incomplete information)
        # add some randomness to simulate real-world user information changes
        self.partial_tag_count = 3

    def get_info(self) -> Dict[str, Any]:
        """get user information summary"""
        # ensure denominator is not zero
        total_tags_count = len(self.all_tags)
        partial_tag_ratio = (
            len(self.partial_tags) / total_tags_count if total_tags_count > 0 else 0
        )

        return {
            "username": self.username,
            "total_tags": total_tags_count,
            "partial_tags": len(self.partial_tags),
            "partial_tag_ratio": partial_tag_ratio,
            "full_profile": self.full_profile,
        }

    @property
    def partial_tags(self) -> List[str]:
        """get partial tags (for testing incomplete information scenarios)"""
        return self.all_tags[: self.partial_tag_count]

    @property
    def tags(self) -> List[str]:
        """default return partial tags"""
        return self.partial_tags


def create_sample_user_profile() -> UserProfile:
    """Create a comprehensive sample user profile for testing progressive learning"""
    full_profile = "choice-driven, high-agency, dominant, protector, strategist; underdog, rivalry, team-vs-team, hero-vs-villain, internal-struggle, tournament conflicts; master-servant, royalty-commoner, captor-captive power-dynamics; high-immersion lore-expander, community-engagement; power-fantasy, moral-ambiguity; isekai escapism; romance, forbidden-love, love-triangles, found-family, reverse-harem; enemies-to-lovers, slow-burn; reincarnation, devil-powers, jujitsu-sorcerer; betrayal, loyalty, survival, redemption; Naruto, Dragon Ball, Jujutsu-Kaisen, Genshin-Impact, One-Piece, Demon-Slayer, Chainsaw-Man, Marvel/DC; crossover, anti-hero, strategy, fan-groups."

    return UserProfile(username="ProgressiveLearningUser", full_profile=full_profile)
