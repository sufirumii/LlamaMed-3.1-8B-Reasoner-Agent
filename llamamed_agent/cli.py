from __future__ import annotations

import argparse
import sys
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from . import guardrails
from .agent.core import Agent
from .agent.parser import AgentStep
from .backends import get_backend
from .config import Config
from .memory import LongTermMemory, SessionMemory
from .rag.ingest import ingest_path
from .tools import build_default_registry

THEME = Theme(
    {
        "user": "bold blue",
        "thought": "dim italic",
        "action": "bold magenta",
        "obs": "grey50",
        "final": "bold green",
        "error": "bold red",
        "warn": "bold yellow",
        "system": "bold white",
        "cmd": "bold cyan",
    }
)
console = Console(theme=THEME)

SLASH_COMMANDS = {
    "/help": "Show this help",
    "/tools": "List available tools",
    "/config": "Show the active configuration",
    "/history": "Show this session's transcript",
    "/sessions": "List all saved sessions",
    "/memory <query>": "Search long-term memory manually",
    "/policy": "Show the agent's guardrail policy",
    "/clear": "Start a new session (keeps long-term memory)",
    "/exit or /quit": "Leave the chat",
}


def _print_step(step: AgentStep, observation: Optional[str]) -> None:
    if step.thought:
        console.print(f"[thought]Thought:[/thought] {step.thought}")
    if observation is None:
        return
    console.print(f"[action]Action:[/action] {step.action}  [action]Input:[/action] {step.action_input}")
    console.print(Panel(observation, title="Observation", border_style="grey50", expand=False))


def _print_help() -> None:
    table = Table(title="Slash commands", show_header=True, header_style="cmd")
    table.add_column("Command")
    table.add_column("Description")
    for cmd, desc in SLASH_COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)


def _print_tools(agent: Agent) -> None:
    console.print(Panel(agent.tools.render_for_prompt(), title="Available tools", border_style="cmd"))


def _print_config(cfg: Config) -> None:
    console.print(Panel(str(cfg), title="Active configuration", border_style="cmd"))


def _build_agent(cfg: Config, session_id: str) -> Agent:
    with console.status("[dim]Loading model...[/dim]"):
        backend = get_backend(cfg.model)

    memory = LongTermMemory(cfg.memory.long_term_dir, cfg.rag.embedding_model) if cfg.memory.enabled else None
    tools = build_default_registry(cfg, backend, memory=memory)
    return Agent(backend, tools, cfg, long_term_memory=memory, session_id=session_id)


def _run_turn(agent: Agent, cfg: Config, query: str, session: Optional[SessionMemory] = None) -> None:
    """Runs one turn, streaming to the console if enabled, then rendering
    the final answer/observations exactly the same way whether or not
    streaming was used.
    """
    on_step = _print_step if cfg.agent.verbose else None

    if cfg.agent.stream:
        live_text = {"buf": ""}

        with Live(console=console, refresh_per_second=12, transient=True) as live:
            def on_token(chunk: str) -> None:
                live_text["buf"] += chunk
                live.update(Panel(live_text["buf"], title="Generating...", border_style="dim"))

            result = agent.run(query, on_step=on_step, on_token=on_token)
    else:
        result = agent.run(query, on_step=on_step)

    if result.blocked:
        console.print(Panel(result.final_answer, title="Blocked by guardrail", border_style="warn"))
    else:
        console.print(Panel(Markdown(result.final_answer), title="Final Answer", border_style="final"))

    if session is not None:
        session.add("user", query)
        session.add("assistant", result.final_answer)


def _handle_slash(cmd: str, agent: Agent, cfg: Config, session: SessionMemory) -> Optional[str]:
    """Returns a new session_id if /clear was used (caller should rebuild
    the agent), or None otherwise. Prints output for every other command.
    """
    parts = cmd.strip().split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if name in ("/help", "/?"):
        _print_help()
    elif name == "/tools":
        _print_tools(agent)
    elif name == "/config":
        _print_config(cfg)
    elif name == "/history":
        console.print(Panel(session.render_history(), title=f"Session {session.session_id}", border_style="cmd"))
    elif name == "/sessions":
        sessions = SessionMemory.list_sessions(cfg.memory.sessions_dir)
        console.print(Panel("\n".join(sessions) or "(none yet)", title="Saved sessions", border_style="cmd"))
    elif name == "/memory":
        if not arg:
            console.print("[warn]Usage: /memory <query>[/warn]")
        elif agent.long_term_memory is None:
            console.print("[warn]Long-term memory is disabled in config.[/warn]")
        else:
            hits = agent.long_term_memory.recall(arg, top_k=5)
            console.print(Panel("\n\n---\n\n".join(hits) or "(no matches)", title="Memory search", border_style="cmd"))
    elif name == "/policy":
        console.print(Panel(guardrails.policy_text(), title="Guardrail policy", border_style="cmd"))
    elif name == "/clear":
        console.print("[system]Starting a new session (long-term memory is preserved).[/system]")
        return "new"
    else:
        console.print(f"[warn]Unknown command '{name}'. Type /help for a list.[/warn]")
    return None


def _make_prompt_session(cfg: Config):
    """Builds a prompt_toolkit session with persistent, arrow-key command
    history. Falls back to plain input() if prompt_toolkit can't attach to
    a real terminal (e.g. some notebook/CI environments) -- streaming and
    slash commands still work either way.
    """
    try:
        from pathlib import Path

        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        hist_path = Path(cfg.memory.history_file)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        return PromptSession(history=FileHistory(str(hist_path)))
    except Exception:
        return None


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config, profile=args.profile)
    if args.backend:
        cfg.model.backend = args.backend
    if args.no_stream:
        cfg.agent.stream = False

    agent = _build_agent(cfg, session_id="one-shot")
    session = SessionMemory(cfg.memory.sessions_dir) if cfg.memory.enabled else None
    _run_turn(agent, cfg, args.query, session=session)


def cmd_chat(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config, profile=args.profile)
    if args.backend:
        cfg.model.backend = args.backend
    if args.no_stream:
        cfg.agent.stream = False

    session_id = args.session
    session = SessionMemory(cfg.memory.sessions_dir, session_id=session_id) if cfg.memory.enabled else SessionMemory(cfg.memory.sessions_dir)
    agent = _build_agent(cfg, session_id=session.session_id)

    console.print(
        Panel(
            f"LlamaMed-3.1-8B-Reasoner-Agent -- session [cmd]{session.session_id}[/cmd]\n"
            "Type a question, or /help for commands, or 'exit'/'quit' to leave.",
            style="system",
        )
    )
    if session.turns:
        console.print(f"[dim]Resumed session with {len(session.turns)} prior turn(s). /history to view.[/dim]")

    prompt_session = _make_prompt_session(cfg)

    while True:
        try:
            if prompt_session is not None:
                query = prompt_session.prompt("you> ")
            else:
                query = console.input("[user]you> [/user]")
        except (EOFError, KeyboardInterrupt):
            break

        query = query.strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "/exit", "/quit"):
            break

        if query.startswith("/"):
            outcome = _handle_slash(query, agent, cfg, session)
            if outcome == "new":
                session = SessionMemory(cfg.memory.sessions_dir)
                agent = _build_agent(cfg, session_id=session.session_id)
            continue

        _run_turn(agent, cfg, query, session=session)


def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config, profile=args.profile)
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
    console.print(f"[final]Done.[/final] Index saved to {cfg.rag.index_dir}")


def cmd_ui(args: argparse.Namespace) -> None:
    cfg = Config.load(args.config, profile=args.profile)
    if args.backend:
        cfg.model.backend = args.backend

    import uvicorn

    from .server import create_app

    with console.status("[dim]Loading model...[/dim]"):
        app = create_app(cfg)

    console.print(Panel(f"Serving at http://127.0.0.1:{args.port}", style="final"))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llamamed-agent")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--profile",
        default=None,
        help="Named profile layered on top of config.yaml, e.g. --profile colab loads config.colab.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="One-shot question")
    ask.add_argument("query")
    ask.add_argument("--backend", choices=["gguf", "transformers"], default=None)
    ask.add_argument("--no-stream", action="store_true", help="Disable token streaming for this run")
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="Interactive chat session")
    chat.add_argument("--backend", choices=["gguf", "transformers"], default=None)
    chat.add_argument("--session", default=None, help="Resume a saved session by id (see /sessions)")
    chat.add_argument("--no-stream", action="store_true", help="Disable token streaming")
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
        console.print(f"[error]Error:[/error] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
