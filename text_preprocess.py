from __future__ import annotations
import re
from typing import List, Dict, Tuple

import spacy
from h2p_parser.h2p import H2p
from g2p_en import G2p

# Load once (safe for a single-process server; if you fork/workers, do this in worker init)
_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
_NLP.add_pipe("sentencizer")
_H2P = H2p(preload=True)
_G2P = G2p()

# ARPAbet phone to grapheme (spelling) map for respelling
# Based on standard correspondences to create unambiguous spellings that TTS should pronounce correctly
_PHONE_MAP: Dict[str, str] = {
    # Updated mappings based on common TTS expectations
    'AA': 'ah',   # "odd" -> "ahd"
    'AE': 'a',    # "at" -> "at"
    'AH': 'uh',   # "hut" -> "huht"
    'AO': 'aw',   # "ought" -> "awt"
    'AW': 'ow',   # "cow" -> "kow"
    'AY': 'ai',   # "high" -> "hai"
    'EH': 'e',    # "ed" -> "ed"
    'ER': 'er',   # "hurt" -> "hert"
    'EY': 'ay',   # "day" -> "day"
    'IH': 'ih',   # "it" -> "iht"
    'IY': 'ee',   # "eat" -> "eet"
    'OW': 'oh',   # "oat" -> "oht"
    'OY': 'oy',   # "toy" -> "toy"
    'UH': 'oo',   # "book" -> "book"
    'UW': 'oo',   # "boot" -> "boot"
    # Consonants (mostly direct)
    'B': 'b',
    'CH': 'ch',
    'D': 'd',
    'DH': 'th',
    'F': 'f',
    'G': 'g',
    'HH': 'h',
    'JH': 'j',
    'K': 'k',
    'L': 'l',
    'M': 'm',
    'N': 'n',
    'NG': 'ng',
    'P': 'p',
    'R': 'r',
    'S': 's',
    'SH': 'sh',
    'T': 't',
    'TH': 'th',
    'V': 'v',
    'W': 'w',
    'Y': 'y',
    'Z': 'z',
    'ZH': 'zh',  # as in "measure"; TTS often handles "zh"
}

# Detect {PHONEMES} chunks produced by h2p.replace_het
_PHONEME_RE = re.compile(r"\{([A-Z0-2 ]+)\}")

def _phone_to_graph(phone: str) -> str:
    """Map ARPAbet phone (without stress) to grapheme."""
    return _PHONE_MAP.get(phone, phone)  # Fallback to phone itself if unknown

def _respell_phones(phones: str, orig: str) -> str:
    """Convert ARPAbet phones to respelled word with validation."""
    phone_list = phones.split()
    graphemes = []
    
    for p in phone_list:
        base_phone = re.sub(r'\d', '', p)
        graph = _PHONE_MAP.get(base_phone)
        if graph is None:
            # Fall back to original word if unknown phone
            return orig
        graphemes.append(graph)
    
    new_word = ''.join(graphemes)
    
    # Validate the respelling has vowels
    if not any(vowel in new_word.lower() for vowel in 'aeiouy'):
        return orig
        
    # Limit length expansion
    if len(new_word) > len(orig) * 1.5:
        return orig
        
    # Preserve case
    if orig.isupper():
        return new_word.upper()
    elif orig.istitle():
        return new_word.capitalize()
    elif orig.islower():
        return new_word.lower()
    else:
        return new_word  # Keep as is for mixed case

def _heteronym_candidates(sent_text: str) -> List[Tuple[int, int, str, str]]:
    """
    Returns a list of (start, end, orig_word, phones) for heteronyms in this sentence.
    Uses deterministic alignment based on token positions in the replaced text.
    """
    doc = _NLP(sent_text)
    repl = _H2P.replace_het(sent_text)
    
    # Get all alpha tokens with their positions
    words = [(t.idx, t.idx + len(t.text), t.text) for t in doc if t.is_alpha]
    
    # Find all {PHONES} in the replaced text
    phone_matches = list(_PHONEME_RE.finditer(repl))
    
    out: List[Tuple[int, int, str, str]] = []
    word_idx = 0
    
    for match in phone_matches:
        phones = match.group(1)
        match_start = match.start()
        
        # Count words (non-{PHONES} tokens) before this match
        pre_text = repl[:match_start]
        # Count words by splitting and excluding {PHONES}
        pre_tokens = [t for t in pre_text.split() if not _PHONEME_RE.fullmatch(t)]
        target_word_idx = len(pre_tokens)
        
        # Find the corresponding word
        if target_word_idx < len(words):
            start, end, orig = words[target_word_idx]
            out.append((start, end, orig, phones))
    
    return out

def _proper_noun_candidates(sent: spacy.tokens.Span) -> List[Tuple[int, int, str, str]]:
    """
    Returns sentence-relative spans for proper nouns with filtering.
    """
    out: List[Tuple[int, int, str, str]] = []
    sent_start = sent.start_char
    
    # Common words that shouldn't be respelled even if tagged as PROPN
    COMMON_PROPN = {"the", "and", "of", "for", "in", "on", "at", "to", "by"}
    
    for t in sent:
        # Expanded check for proper nouns with punctuation
        if (t.pos_ == "PROPN" and 
            re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", t.text) and
            t.text.lower() not in COMMON_PROPN and
            len(t.text) >= 3):  # Skip very short proper nouns
            
            phones_list = _G2P(t.text.lower())
            phones = ' '.join(phones_list)
            out.append((t.idx - sent_start, t.idx - sent_start + len(t.text), t.text, phones))
    
    return out

def preprocess_text(text: str) -> str:
    """
    Preprocess text: 
    1) Split into sentences (spaCy).
    2) For each sentence, detect and respell heteronyms (context-aware via h2p).
    3) Detect and respell proper nouns (names/locations via spaCy POS and g2p-en).
    Prioritize heteronym respellings if overlap.
    """
    try:
        doc = _NLP(text)
        new_chunks: List[str] = []
        last_end = 0

        for sent in doc.sents:
            sent_text = sent.text
            sent_start = sent.start_char

            # Append any non-sentence gap
            if last_end < sent_start:
                new_chunks.append(text[last_end:sent_start])

            # Collect heteronym replacements (relative to sent)
            het_rel = _heteronym_candidates(sent_text)
            het_repls = [(sent_start + s, sent_start + e, o, ph) for s, e, o, ph in het_rel]

            # Collect proper noun candidates (relative to sent)
            propn_rel = _proper_noun_candidates(sent)
            propn_repls = [(sent_start + s, sent_start + e, o, ph) for s, e, o, ph in propn_rel]

            # Filter propn if overlaps with het
            het_spans = [(s, e) for s, e, _, _ in het_repls]
            filtered_propn = []
            for p_s, p_e, p_orig, p_phones in propn_repls:
                if not any(h_s < p_e and h_e > p_s for h_s, h_e in het_spans):
                    filtered_propn.append((p_s, p_e, p_orig, p_phones))

            # Combine, sort by start
            all_repls = het_repls + filtered_propn
            all_repls.sort(key=lambda x: x[0])

            if not all_repls:
                new_chunks.append(sent_text)
                last_end = sent_start + len(sent_text)
                continue

            # Build sentence with replacements (handle no overlaps)
            cursor = 0
            built = []
            for abs_start, abs_end, orig, phones in all_repls:
                rel_start = abs_start - sent_start
                rel_end = abs_end - sent_start
                # Add text before this replacement
                built.append(sent_text[cursor:rel_start])
                # Add respelled word
                new_word = _respell_phones(phones, orig)
                built.append(new_word)
                cursor = rel_end

            # Add the rest of the sentence
            built.append(sent_text[cursor:])
            new_chunks.append("".join(built))
            last_end = sent_start + len(sent_text)

        # Tail after last sentence
        if last_end < len(text):
            new_chunks.append(text[last_end:])

        return "".join(new_chunks)
    except Exception as e:
        # Fallback to original on any error
        import logging
        logging.warning(f"preprocess_text failed: {str(e)}")
        return text