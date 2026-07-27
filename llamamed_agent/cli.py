from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agent.core import Agent
from .agent.parser import AgentStep
from .backends import get_backend
from .config import Config
from .rag.ingest import ingest_path
from .tools import build_default_registry

console = Console()


def _print_step(step: AgentStep, observation: str | None) -> None:
    if step.thought:
        console.print(f"[dim]Thought:[/dim] {step.thought}")
    if observation is None:
        return
    console.print(f"[cyan]Action:[/cyan] {step.action}  [cyan]Input:[/cyan] {step.action_input}")
    console.print(Panel(observation, title="Observation", border_style="grey50", expand=False))


def _build_agent(cfg: Config) -> Agent:
    with console.status("[dim]Loading model...[/dim]"):
        backend = get_backend(cfg.model)
    tools = build_default_registry(cfg, backend)
    return Agent(backend, tools, cfg)


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    if args.backend:
        cfg.model.backend = args.backend
    agent = _build_agent(cfg)

    on_step = _print_step if cfg.agent.verbose else None
    result = agent.run(args.query, on_step=on_step)
    console.print(Panel(Markdown(result.final_answer), title="Final Answer", border_style="green"))


def cmd_chat(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    if args.backend:
        cfg.model.backend = args.backend
    agent = _build_agent(cfg)

    console.print(Panel("LlamaMed-3.1-8B-Reasoner-Agent -- type 'exit' to quit", style="bold"))
    on_step = _print_step if cfg.agent.verbose else None

    while True:
        try:
            query = console.input("[bold blue]you> [/bold blue]")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("exit", "quit"):
            break
        if not query.strip():
            continue

        result = agent.run(query, on_step=on_step)
        console.print(Panel(Markdown(result.final_answer), title="Final Answer", border_style="green"))


def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    with console.status(f"[dim]Ingesting {args.path}...[/dim]"):
        results = ingest_path(
            args.path,
            index_dir=cfg.rag.index_dir,
            embedding_model=cfg.rag.embedding_model,
            chunk_size=cfg.rag.chunk_size,
            chunk_overlap=cfg.rag.chunk_overlap,
        )
    for name, n_chunks in results.items():
        console.print(f"  {name}: {n_chunks} chunks")
    console.print(f"[green]Done.[/green] Index saved to {cfg.rag.index_dir}")


def cmd_ui(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config)
    if args.backend:
        cfg.model.backend = args.backend

    import uvicorn

    from .server import create_app

    with console.status("[dim]Loading model...[/dim]"):
        app = create_app(cfg)

    console.print(Panel(f"Serving at http://127.0.0.1:{args.port}", style="bold green"))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llamamed-agent")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="One-shot question")
    ask.add_argument("query")
    ask.add_argument("--backend", choices=["gguf", "transformers"], default=None)
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="Interactive chat session")
    chat.add_argument("--backend", choices=["gguf", "transformers"], default=None)
    chat.set_defaults(func=cmd_chat)

    ingest = sub.add_parser("ingest", help="Ingest a PDF or a directory of PDFs")
    ingest.add_argument("path")
    ingest.set_defaults(func=cmd_ingest)

    ui = sub.add_parser("ui", help="Launch the local web UI")
    ui.add_argument("--backend", choices=["gguf", "transformers"], default=None)
    ui.add_argument("--port", type=int, default=8000)
    ui.set_defaults(func=cmd_ui)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:  # noqa: BLE001 - top-level CLI error boundary
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
