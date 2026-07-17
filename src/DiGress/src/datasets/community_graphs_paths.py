"""Path helpers for CommunityGraphs (stdlib only; safe to test without torch)."""
import os


def resolve_community_graphs_pickle(base_path: str, pickle_rel: str) -> str:
    """Resolve ``pickle_rel`` against DiGress repo root (``base_path``).

    In the OGAD monorepo, data often lives at ``<ogad>/data`` while DiGress sits at
    ``<ogad>/src/vanilla/DiGress``, so configs use ``../../../data/train/graphs/...``.

    In Docker, only the DiGress tree is copied to ``/app`` and OGAD data is mounted at
    ``/app/data``; the same ``../../../data/...`` would normalize to ``/data/...`` (wrong).
    If the primary path is missing, try ``<base_path>/data/train/graphs/<basename>``.
    """
    base_path = os.path.normpath(str(base_path))
    primary = os.path.normpath(os.path.join(base_path, pickle_rel))
    if os.path.isfile(primary):
        return primary
    alt = os.path.normpath(
        os.path.join(base_path, 'data', 'train', 'graphs', os.path.basename(pickle_rel))
    )
    if os.path.isfile(alt):
        return alt
    raise FileNotFoundError(
        f'CommunityGraphs pickle not found. Tried:\n  {primary}\n  {alt}'
    )
