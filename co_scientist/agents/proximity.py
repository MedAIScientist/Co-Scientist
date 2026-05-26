"""Proximity agent — embeds + clusters hypotheses. No LLM call.

The dedup-on-save path is handled inline by Generation/Evolution agents (they
need the FAISS read to skip duplicates before persisting). This agent owns the
*batch* recluster: it scans hypotheses that don't yet have an embedding,
embeds them, refreshes the index on disk, then runs agglomerative clustering
on the cosine distance matrix to refresh `hypotheses.dedup_cluster`.
"""

from __future__ import annotations

import numpy as np

from .. import ids
from ..logging import get_logger
from ..models import Task, TaskResult
from ..storage.repos import embeddings as emb_repo
from ..storage.repos import hypotheses as hyp_repo
from ..vectors.embedder import make_embedder
from ..vectors.store import FaissStore
from .base import BaseAgent

log = get_logger("proximity")


class ProximityAgent(BaseAgent):
    name = "proximity"

    async def execute(self, task: Task) -> TaskResult:
        rebuild = bool(task.payload.get("rebuild", False))
        session_id = task.session_id

        try:
            embedder = make_embedder(self.deps.cfg)
        except (RuntimeError, ValueError) as e:
            log.warning("proximity_no_embedder", err=str(e))
            return TaskResult(kind="noop", extra={"reason": "no embedder configured"})

        store = FaissStore(self.deps.cfg, session_id, dim=embedder.dim)
        await store.load_or_create()

        hyps = await hyp_repo.list_for_session(self.deps.db, session_id)
        # Identify hypotheses missing an embedding for this model. One query
        # over the whole session instead of N has_embedding() probes.
        rows = await emb_repo.list_for_session(self.deps.db, session_id)
        existing_for_model = {
            r["hypothesis_id"] for r in rows if r["model"] == embedder.model
        }
        to_embed = [h for h in hyps if h.id not in existing_for_model]

        if to_embed:
            texts = [(h.title + "\n\n" + h.summary) for h in to_embed]
            vecs = await embedder.embed(texts)
            for h, v in zip(to_embed, vecs, strict=True):
                offset = await store.add(h.id, v)
                await emb_repo.upsert(
                    self.deps.db,
                    id_=ids.embedding_id(h.id, embedder.model),
                    session_id=session_id,
                    hypothesis_id=h.id,
                    model=embedder.model,
                    dim=embedder.dim,
                    faiss_offset=offset,
                    text_hash=ids.text_hash(h.title + "\n\n" + h.summary),
                )
            await store.save()
            log.info("proximity_embedded", n_new=len(to_embed))

        n_indexed = store.n
        clusters_set = False
        if rebuild and n_indexed >= 3:
            try:
                clusters = await self._cluster(store, threshold=self.deps.cfg.vectors.cluster_threshold)
            except Exception as e:
                log.warning("proximity_cluster_failed", err=str(e))
                clusters = None
            if clusters is not None:
                for hid, cluster_id in clusters.items():
                    await hyp_repo.set_dedup_cluster(self.deps.db, hid, cluster_id)
                clusters_set = True
                log.info("proximity_clustered", n_clusters=len(set(clusters.values())), n_indexed=n_indexed)

        return TaskResult(
            kind="proximity_updated",
            extra={"n_indexed": n_indexed, "n_new": len(to_embed), "clustered": clusters_set},
        )

    # ----------------------------- clustering ----------------------------- #

    async def _cluster(
        self, store: FaissStore, *, threshold: float
    ) -> dict[str, str] | None:
        """Run sklearn agglomerative clustering on the cosine distance matrix.

        Returns {hypothesis_id: cluster_label}. Threshold is the maximum cosine
        *distance* (1 - cosine_similarity) that triggers merging.
        """
        import asyncio

        from sklearn.cluster import AgglomerativeClustering

        n = store.n
        if n < 2 or store.index is None:
            return None
        sim = await store.cosine_matrix()    # locked + threaded
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        # symmetrize numerically
        dist = (dist + dist.T) / 2.0
        np.clip(dist, 0.0, 2.0, out=dist)

        def _do() -> np.ndarray:
            ac = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=threshold,
            )
            return ac.fit_predict(dist)

        labels = await asyncio.to_thread(_do)
        out: dict[str, str] = {}
        for i, label in enumerate(labels):
            hid = store.hypothesis_at(i)
            if hid is None:
                continue
            out[hid] = f"c{int(label):04d}"
        return out
