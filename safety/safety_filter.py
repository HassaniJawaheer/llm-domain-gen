import re

# phrase matches
BLOCK_PHRASES = [
    # EN
    "adult content", "adult website", "adult platform", "18+ content",
    "explicit content", "explicit photos", "explicit videos",
    "nude content", "nudity content", "nsfw content",
    "subscription-based adult", "premium adult content",
    "onlyfans", "fansly", "fanvue",
    "cam site", "live cam", "webcam model", "webcam modeling",
    "lingerie modeling", "glamour modeling nude",
    "erotic massage", "sensual massage",
    "escort agency", "companionship services",
    "sugar dating", "sugar baby", "sugar daddy",
    "gentlemen's club", "strip club",
    "fetish content", "kink content", "bdsm content",
    "sex toys", "adult toys", "sex shop",
    "porn site", "pornographic content", "xxx videos",
]

# proximity patterns
_CONTENT_WORDS = r"(website|site|platform|service|subscription|store|shop|marketplace|channel|account|page|app|application|streaming|media|content|video(?:s)?|photo(?:s)?|images?|pics?|gallery)"
_TRIGGER = r"(adult|explicit|nude|nudity|nsfw|xxx|porn(?:ographic)?|erotic(?:a)?|sexual|sex|fetish|kink|bdsm|cam)"

BLOCK_REGEX = [
    # EN
    re.compile(rf"\b{_TRIGGER}\b(?:\W+\w+){{0,3}}\W+\b{_CONTENT_WORDS}\b", re.IGNORECASE),
    re.compile(rf"\b{_CONTENT_WORDS}\b(?:\W+\w+){{0,3}}\W+\b{_TRIGGER}\b", re.IGNORECASE),
]

def is_inappropriate(text: str) -> bool:
    tl = (text or "").lower()
    for p in BLOCK_PHRASES:
        if p in tl:
            return True
    for rx in BLOCK_REGEX:
        if rx.search(text or ""):
            return True
    return False
