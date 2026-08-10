"""Orchestration hub: ingest, retrieve, route, and the remote cost guard.

The CLI/REPL is a thin shell over this. All remote (paid) calls go through
`run_remote`, which enforces the pre-call cost estimate, confirmation
threshold, and per-session budget cap.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from docmind.config import IMAGES_DIR, Config
from docmind.ingest import chunker, diagrams, parsers
from docmind.llm import pricing
from docmind.llm.local import LocalLLM
from docmind.llm.provider import Completion, Provider, get_provider
from docmind.llm.router import Route, route_for_ask
from docmind.store.embeddings import Embedder
from docmind.store.manifest import Manifest, doc_id_for, new_record
from docmind.store.vectordb import Hit, VectorStore

# callbacks the CLI supplies for UI
Logger = Callable[[str], None]
Confirmer = Callable[[str], bool]   # message -> proceed?


class BudgetExceeded(Exception):
    pass


class Aborted(Exception):
    pass


@dataclass
class IngestResult:
    doc_id: str
    title: str
    chunk_count: int
    diagram_count: int
    skipped: bool = False


@dataclass
class RemoteEstimate:
    provider: str
    model: str
    input_tokens: int
    est_usd: float
    available: bool
    reason: str
    over_cap: bool


@dataclass
class GenResult:
    text: str
    route: str                 # "local" | "remote"
    provider: str | None
    model: str | None
    cost: float


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = Embedder(cfg)
        self.manifest = Manifest()
        self._store: VectorStore | None = None
        self.session_spent = 0.0

    # --- lazy store -------------------------------------------------------
    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(self.cfg)  # opens existing table if present
        return self._store

    # --- ingest -----------------------------------------------------------
    def ingest_path(self, path: Path, log: Logger, force: bool = False) -> list[IngestResult]:
        path = path.expanduser()
        targets: list[Path] = []
        if path.is_dir():
            for p in sorted(path.rglob("*")):
                if p.suffix.lower() in parsers.SUPPORTED:
                    targets.append(p)
        elif path.suffix.lower() in parsers.SUPPORTED:
            targets.append(path)
        else:
            raise ValueError(f"Unsupported or missing path: {path}")

        results: list[IngestResult] = []
        for p in targets:
            results.append(self._ingest_file(p, log, force))
        return results

    def _ingest_file(self, path: Path, log: Logger, force: bool) -> IngestResult:
        doc_id = doc_id_for(path)
        if not force and self.manifest.unchanged(path):
            log(f"  skip (unchanged): {path.name}")
            rec = self.manifest.get(doc_id)
            return IngestResult(doc_id, rec.source_path if rec else path.stem,
                                rec.chunk_count if rec else 0,
                                rec.diagram_count if rec else 0, skipped=True)

        # replace any prior version cleanly
        self.remove(doc_id, log=lambda m: None, silent=True)

        log(f"  parsing: {path.name}")
        parsed = parsers.parse(path, doc_id)

        # describe diagrams with the local vision model
        diagram_descs: list[tuple[str, int, str]] = []
        for img in parsed.images:
            desc = diagrams.describe_image(self.cfg, img.path)
            if desc:
                diagram_descs.append((desc, img.page, img.section))
        if parsed.images:
            log(f"  diagrams: {len(diagram_descs)}/{len(parsed.images)} images described")

        chunks = chunker.build_chunks(self.cfg, doc_id, str(path.resolve()), parsed, diagram_descs)
        if not chunks:
            log(f"  (no text extracted) {path.name}")
            return IngestResult(doc_id, parsed.title, 0, 0)

        log(f"  embedding {len(chunks)} chunks…")
        vectors = self.embedder.embed([c.text for c in chunks])
        for c, v in zip(chunks, vectors):
            c.vector = v

        self.store.add(chunks)
        rec = new_record(path, chunk_count=len(chunks), diagram_count=len(diagram_descs))
        self.manifest.upsert(rec)
        log(f"  indexed: {path.name} ({len(chunks)} chunks)")
        return IngestResult(doc_id, parsed.title, len(chunks), len(diagram_descs))

    # --- remove -----------------------------------------------------------
    def remove(self, doc_id: str, log: Logger, silent: bool = False):
        self.store.delete_doc(doc_id)
        img_dir = IMAGES_DIR / doc_id
        if img_dir.exists():
            shutil.rmtree(img_dir, ignore_errors=True)
        rec = self.manifest.remove(doc_id)
        if rec and not silent:
            log(f"removed: {rec.source_path}")
        return rec

    # --- retrieve ---------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None, kind: str | None = None) -> list[Hit]:
        vec = self.embedder.embed_one(query)
        return self.store.search(vec, k or self.cfg.top_k, kind=kind)

    # --- local answer -----------------------------------------------------
    def answer_local(self, system: str, user: str) -> str:
        return LocalLLM(self.cfg).chat(system, user)

    def answer_local_stream(self, system: str, user: str):
        """Yield answer chunks as they arrive (for responsive UIs)."""
        yield from LocalLLM(self.cfg).chat_stream(system, user)

    # --- remote estimate (no call) ---------------------------------------
    def estimate_remote(
        self, system: str, user: str, *, heavy: bool, provider_name: str | None = None,
        output_guess: int = 1500,
    ) -> "RemoteEstimate":
        """Cost/availability preview so a UI can show the price before committing."""
        provider: Provider = get_provider(self.cfg, provider_name)
        available, reason = provider.available()
        model = provider.model_for(heavy)
        in_tokens = provider.count_tokens(system, user, heavy=heavy) if available else 0
        est = provider.estimate(in_tokens, output_guess, heavy=heavy) if available else 0.0
        return RemoteEstimate(
            provider=provider.name,
            model=model,
            input_tokens=in_tokens,
            est_usd=est,
            available=available,
            reason=reason,
            over_cap=self.session_spent + est > self.cfg.budget_cap,
        )

    # --- remote (paid) with cost guard -----------------------------------
    def run_remote(
        self,
        system: str,
        user: str,
        *,
        heavy: bool,
        log: Logger,
        confirm: Confirmer,
        provider_name: str | None = None,
        output_guess: int = 1500,
        max_tokens: int = 4096,
    ) -> Completion:
        provider: Provider = get_provider(self.cfg, provider_name)
        ok, msg = provider.available()
        if not ok:
            raise Aborted(msg)

        in_tokens = provider.count_tokens(system, user, heavy=heavy)
        est = provider.estimate(in_tokens, output_guess, heavy=heavy)
        model = provider.model_for(heavy)
        log(f"→ {provider.name}/{model}: ~{in_tokens} in tokens, est ${est:.4f}")

        if self.session_spent + est > self.cfg.budget_cap:
            if not confirm(
                f"Estimated ${est:.4f} would exceed session budget cap "
                f"(${self.cfg.budget_cap:.2f}; spent ${self.session_spent:.4f}). Proceed?"
            ):
                raise BudgetExceeded("budget cap reached")
        elif est >= self.cfg.confirm_threshold:
            if not confirm(f"This call is estimated at ${est:.4f}. Proceed?"):
                raise Aborted("declined at cost prompt")

        comp = provider.complete(system, user, heavy=heavy, max_tokens=max_tokens)
        actual = pricing.estimate_cost(
            comp.provider,
            comp.model,
            comp.usage.input_tokens,
            comp.usage.output_tokens,
            comp.usage.cache_read_tokens,
            comp.usage.cache_write_tokens,
        )
        self.session_spent += actual
        cache_note = (
            f", cache_read={comp.usage.cache_read_tokens}"
            if comp.usage.cache_read_tokens
            else ""
        )
        log(
            f"✓ {comp.provider}/{comp.model}: "
            f"in={comp.usage.input_tokens} out={comp.usage.output_tokens}{cache_note} "
            f"cost≈${actual:.4f} (session ${self.session_spent:.4f})"
        )
        return comp

    # --- generate: remote if possible, else local ------------------------
    def generate(
        self,
        system: str,
        user: str,
        *,
        heavy: bool,
        log: Logger,
        confirm: Confirmer,
        force_local: bool = False,
        provider_name: str | None = None,
        output_guess: int = 1500,
        max_tokens: int = 4096,
    ) -> GenResult:
        """Non-streaming generation used by diagram/gaps/critique.

        Tries the remote provider unless `force_local`; on unavailability,
        budget cap, or a declined cost prompt it falls back to the local model
        so these features still work offline / without a key.
        """
        if not force_local:
            try:
                comp = self.run_remote(
                    system, user, heavy=heavy, log=log, confirm=confirm,
                    provider_name=provider_name, output_guess=output_guess,
                    max_tokens=max_tokens,
                )
                cost = pricing.estimate_cost(
                    comp.provider, comp.model,
                    comp.usage.input_tokens, comp.usage.output_tokens,
                    comp.usage.cache_read_tokens, comp.usage.cache_write_tokens,
                )
                return GenResult(comp.text, "remote", comp.provider, comp.model, cost)
            except (Aborted, BudgetExceeded) as e:
                log(f"remote unavailable ({e}); using local model")

        text = self.answer_local(system, user)
        return GenResult(text, "local", None, self.cfg.local_chat_model, 0.0)

    # --- routing helper ---------------------------------------------------
    def route(self, question: str, force: Route | None = None) -> Route:
        return route_for_ask(question, force)
