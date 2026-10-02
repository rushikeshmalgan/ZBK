"""Fast player search — a Trie (prefix tree) autocomplete index.

WHY A TRIE?
    The naive approach filters the whole DataFrame on every keystroke: O(N x L) per query
    (N = 1,826 players, L = query length), repeated on each character typed. That re-scans
    everyone every time. A Trie answers a prefix query in O(L + M) — walk L nodes to the prefix,
    then collect the M names beneath it — independent of N. A hash map gives O(1) exact lookup.

DATA STRUCTURES
    1. Trie / prefix tree  -> prefix autocomplete in O(L + M)
    2. Hash map (dict)     -> exact name -> row index in O(1) average
    Both are built ONCE at app start (cached) from the player names; typing never rebuilds them.

INDEXING
    Each player's full normalised name AND each of its tokens (e.g. "bukayo", "saka") are inserted,
    so a surname prefix ("sak") finds "Bukayo Saka". Accents are stripped so "oscar" matches "Óscar".

COMPLEXITY
    Build : O(total characters across all names)      space O(total characters)
    Prefix: O(L + M)   (L = len(query), M = matches)  -> top-K sorted by score in O(M log K)
    Exact : O(1) average via the hash map
    This is the efficient, DSA-backed search requested for the project demo.
"""
import unicodedata


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower().strip()


class _Node:
    __slots__ = ("children", "ids")

    def __init__(self):
        self.children = {}
        self.ids = set()          # row indices for every name/token passing through this node


class PlayerSearchIndex:
    def __init__(self, names, scores):
        self._root = _Node()
        self._exact = {}                       # hash map: exact normalised name -> row index
        self._score = {}                       # row index -> score (for ranking top-K)
        for idx, (name, score) in enumerate(zip(names, scores)):
            n = _norm(name)
            self._exact[n] = idx
            self._score[idx] = float(score)
            for token in [n] + n.split():      # index the full name and each token
                self._insert(token, idx)

    def _insert(self, word, idx):
        node = self._root
        for ch in word:
            node = node.children.setdefault(ch, _Node())
            node.ids.add(idx)

    def lookup(self, name):
        """O(1) average exact lookup -> row index or None."""
        return self._exact.get(_norm(name))

    def search(self, query, limit=5):
        """Prefix search -> up to `limit` row indices, best score first. O(L + M) + O(M log K)."""
        q = _norm(query)
        if not q:
            return []
        node = self._root
        for ch in q:                           # O(L): walk to the prefix node
            node = node.children.get(ch)
            if node is None:
                return []                      # no player matches this prefix
        ids = node.ids                         # O(1): all matches live on this node
        return sorted(ids, key=lambda i: self._score.get(i, 0.0), reverse=True)[:limit]


def build_index(df):
    """Build the index once from a players DataFrame (expects Player + Score columns)."""
    return PlayerSearchIndex(df["Player"].tolist(), df["Score"].tolist())
