# zipvoice/text_preprocess.py
from __future__ import annotations
import re
from typing import List, Dict, Tuple

import spacy
from h2p_parser.h2p import H2p
from g2p_en import G2p

# Load once (safe for a single-process server; if you fork/workers, do this in worker init)
_NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
_H2P = H2p(preload=True)
_G2P = G2p()

# ARPAbet phone to grapheme (spelling) map for respelling
# Based on standard correspondences to create unambiguous spellings that TTS should pronounce correctly
_PHONE_MAP: Dict[str, str] = {
    # Vowels (chosen to force common pronunciations in English TTS)
    'AA': 'o',   # as in "odd"
    'AE': 'a',   # as in "at"
    'AH': 'u',   # as in "hut"
    'AO': 'aw',  # as in "ought"
    'AW': 'ow',  # as in "cow"
    'AY': 'igh', # as in "high"
    'EH': 'e',   # as in "ed"
    'ER': 'ur',  # as in "hurt"
    'EY': 'ay',  # as in "day"
    'IH': 'i',   # as in "it"
    'IY': 'ee',  # as in "eat"
    'OW': 'o',   # as in "oat"
    'OY': 'oy',  # as in "toy"
    'UH': 'oo',  # as in "book"
    'UW': 'oo',  # as in "boot"
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
    """Convert ARPAbet phones string to respelled word, preserving case."""
    phone_list = phones.split()
    graphemes = [_phone_to_graph(re.sub(r'\d', '', p)) for p in phone_list]
    new_word = ''.join(graphemes)
    # Preserve original case
    if orig.isupper():
        new_word = new_word.upper()
    elif orig.istitle():
        new_word = new_word.capitalize()
    elif orig.islower():
        new_word = new_word.lower()
    else:
        new_word = new_word.capitalize()  # Default for mixed case
    return new_word

def _heteronym_candidates(sent_text: str) -> List[Tuple[int, int, str, str]]:
    """
    Returns a list of (start, end, orig_word, phones) for heteronyms in this sentence,
    using h2p to determine the *contextual* pronunciation.
    """
    doc = _NLP(sent_text)

    # Build a simple token stream (indices and text) for words only
    words = [(t.idx, t.idx + len(t.text), t.text, t.lemma_.lower()) for t in doc if t.is_alpha]

    # h2p: replace only heteronyms with {PHONES}
    repl = _H2P.replace_het(sent_text)

    # Extract just the {PHONES} in order
    phones_in_order = _PHONEME_RE.findall(repl)

    out: List[Tuple[int, int, str, str]] = []
    wi = 0  # index over 'words'
    for ph in phones_in_order:
        # Advance to next word (assuming h2p replaces in token order)
        while wi < len(words) and words[wi][3] not in _H2P.heteronyms:  # Optional: check if known het
            wi += 1
        if wi >= len(words):
            break
        start, end, orig, lemma = words[wi]
        out.append((start, end, orig, ph))
        wi += 1
    return out

def _proper_noun_candidates(sent: spacy.tokens.Span) -> List[Tuple[int, int, str, str]]:
    """
    Returns a list of (start, end, orig_word, phones) for proper nouns (PROPN) in this sentence,
    using g2p-en to predict phones.
    """
    out: List[Tuple[int, int, str, str]] = []
    for t in sent:
        if t.pos_ == "PROPN" and t.is_alpha:  # Proper nouns, alphabetic only
            phones_list = _G2P(t.text.lower())  # g2p-en takes lowercase
            phones = ' '.join(phones_list)
            out.append((t.idx, t.idx + len(t.text), t.text, phones))
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
            het_repls = _heteronym_candidates(sent_text)

            # Collect proper noun candidates (relative to sent)
            propn_repls = _proper_noun_candidates(sent)

            # Filter propn if overlaps with het
            het_spans = [(s, e) for s, e, _, _ in het_repls]
            filtered_propn = []
            for p_s, p_e, p_orig, p_phones in propn_repls:
                if not any(h_s < p_e and h_e > p_s for h_s, h_e in het_spans):
                    filtered_propn.append((p_s, p_e, p_orig, p_phones))

            # Combine, make absolute starts, sort by start
            all_repls = [
                (sent_start + s, sent_start + e, o, ph) for s, e, o, ph in het_repls + filtered_propn
            ]
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