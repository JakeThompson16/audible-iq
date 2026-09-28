"""
Audible IQ player lookup: search a player, see their projected stat line.

Run (from the repo root):   python webapp/main.py      -> http://localhost:8090
                            (python webapp/main.py --port 9000 to use another port)

Presentation only. Suggestions come from search.player_search.PlayerIndex.suggest
and projections from search.projection_service.project_player; this file has
no autocomplete, modeling or data logic of its own (labels and display
rounding only).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `python webapp/main.py`

from nicegui import app, run, ui  # noqa: E402

from pipeline.artifacts import read_all  # noqa: E402
from search.player_search import build_player_index  # noqa: E402
from search.projection_service import project_player  # noqa: E402

DEBOUNCE_S = 0.2
DEFAULT_PORT = 8090  # 8080 (NiceGUI's default) is often taken; override with --port

# Display order and labels per position (presentation only).
STAT_LAYOUT = {
    "QB": [("attempts", "Pass att"), ("completions", "Completions"), ("passing_yards", "Pass yds"),
           ("passing_tds", "Pass TD"), ("passing_interceptions", "INT"), ("carries", "Rush att"),
           ("rushing_yards", "Rush yds"), ("rushing_tds", "Rush TD")],
    "RB": [("carries", "Rush att"), ("rushing_yards", "Rush yds"), ("rushing_tds", "Rush TD"),
           ("targets", "Targets"), ("receptions", "Receptions"), ("receiving_yards", "Rec yds"),
           ("receiving_tds", "Rec TD")],
}
STAT_LAYOUT["WR"] = STAT_LAYOUT["TE"] = [
    ("targets", "Targets"), ("receptions", "Receptions"), ("receiving_yards", "Rec yds"),
    ("receiving_tds", "Rec TD"), ("carries", "Rush att"), ("rushing_yards", "Rush yds"), ("rushing_tds", "Rush TD"),
]

STATUS_MESSAGES = {
    "bye": ("event_busy", "Bye week", "{team} has no game in week {week} ({season})."),
    "no_history": ("person_search", "No projection yet",
                   "No NFL games before this week (rookie or no history), so there's nothing to project from."),
    "no_game_scheduled": ("event_note", "No game scheduled", "{reason}"),
    "unsupported_position": ("block", "Position not supported", "Projections cover QB, RB, WR and TE only."),
    "unknown_player": ("help", "Unknown player", "{reason}"),
}

# Process-wide, read-only data shared by all visitors (loaded once at startup).
STATE = {"index": None, "model_info": None, "error": None}


def _load_data() -> None:
    """Blocking: feature context + player index + artifact metadata (runs in a worker thread)."""
    STATE["index"] = build_player_index()
    arts = read_all()
    STATE["model_info"] = next(iter(arts.values()), None)


async def _startup() -> None:
    try:
        await run.io_bound(_load_data)
    except Exception as exc:  # surfaced on the page instead of a silent blank screen
        STATE["error"] = repr(exc)


app.on_startup(_startup)


def _fmt(x) -> str:
    return "–" if x is None else f"{x:.1f}"


@ui.page("/")
def index_page() -> None:
    ui.page_title("Audible IQ · Player projections")
    selection = {"result": None, "gsis_id": None}
    pending = {"timer": None, "suppress": False}

    with ui.column().classes("w-full max-w-2xl mx-auto p-4 gap-3"):
        ui.label("Player projections").classes("text-2xl font-bold")

        loading = ui.row().classes("items-center gap-2").mark("loading")
        with loading:
            ui.spinner(size="md")
            ui.label("Loading player data and models…")

        main = ui.column().classes("w-full gap-3")
        main.set_visibility(False)
        with main:
            search = ui.input(placeholder="Search a player (e.g. Mahomes)").props("clearable outlined autofocus") \
                .classes("w-full").mark("search")
            suggestions = ui.column().classes("w-full gap-0").mark("suggestions")
            dev = ui.switch("Dev mode", value=False).mark("dev-toggle")
            ui.label("Injury / inactive status is not modeled: a player who is ruled out still gets a projection.") \
                .classes("text-xs text-gray-500").mark("injury-note")
            card_area = ui.column().classes("w-full")

        footer = ui.label("").classes("text-xs text-gray-500").mark("model-info")

    # ---- suggestions (debounced) -------------------------------------------------
    def show_suggestions(query: str) -> None:
        suggestions.clear()
        if not query or STATE["index"] is None:
            return
        hits = STATE["index"].suggest(query, size=8)
        with suggestions:
            if not hits:
                ui.label("No matching current-season QB/RB/WR/TE.").classes("text-sm text-gray-500")
                return
            with ui.list().props("bordered separator").classes("w-full"):
                for s in hits:
                    ui.item(s["display"], on_click=lambda _e, s=s: select(s["gsis_id"])) \
                        .classes("cursor-pointer").mark("suggestion")

    def on_search(e) -> None:
        if pending["timer"] is not None:
            pending["timer"].cancel()
        if pending["suppress"]:          # the box was filled by a selection, not by typing
            pending["suppress"] = False
            return
        query = e.value or ""
        pending["timer"] = ui.timer(DEBOUNCE_S, lambda: show_suggestions(query), once=True)

    search.on_value_change(on_search)

    # ---- projection -----------------------------------------------------------
    async def select(gsis_id: str, season: int | None = None, week: int | None = None) -> None:
        suggestions.clear()
        entry = STATE["index"].get(gsis_id)
        if entry and search.value != entry["name"]:
            pending["suppress"] = True
            search.set_value(entry["name"])
        selection["gsis_id"] = gsis_id
        selection["result"] = await run.io_bound(project_player, gsis_id, season, week)
        render_card.refresh()

    @ui.refreshable
    def render_card() -> None:
        r = selection["result"]
        if r is None:
            return
        player, target = r["player"], r.get("target") or {}
        with ui.card().classes("w-full").mark("result-card"):
            ui.label(player.get("name") or "Unknown").classes("text-xl font-semibold")
            ui.label(" · ".join(x for x in (player.get("position"), player.get("team")) if x)).classes("text-gray-600")
            if target.get("week"):
                where = f" ({target['home_away']})" if target.get("home_away") else ""
                opp = target.get("opponent") or "—"
                ui.label(f"Week {target['week']} ({target['season']}) vs {opp}{where}") \
                    .classes("text-lg font-medium").mark("game-header")

            if r["status"] != "ok":
                icon, title, text = STATUS_MESSAGES.get(r["status"], ("info", r["status"], "{reason}"))
                with ui.row().classes("items-center gap-2").mark("status-message"):
                    ui.icon(icon, size="md")
                    ui.label(title).classes("font-semibold")
                ui.label(text.format(team=player.get("team"), week=target.get("week"),
                                     season=target.get("season"), reason=r.get("reason") or "")) \
                    .classes("text-gray-700")
            else:
                show_actual = dev.value and target.get("game_final") and r.get("actual")
                with ui.grid(columns=3 if show_actual else 2).classes("w-full gap-x-6 gap-y-1").mark("stat-line"):
                    ui.label("Stat").classes("text-gray-500")
                    ui.label("Projected").classes("text-gray-500")
                    if show_actual:
                        with ui.row().classes("items-center gap-1"):
                            ui.label("Actual").classes("text-gray-500")
                            if r.get("in_training_window"):
                                ui.badge("in-sample", color="orange").mark("in-sample")
                    for key, label in STAT_LAYOUT[player["position"]]:
                        if key not in r["stats"]:
                            continue
                        ui.label(label)
                        ui.label(_fmt(r["stats"][key])).classes("font-mono")
                        if show_actual:
                            ui.label(_fmt(r["actual"].get(key))).classes("font-mono")
                if dev.value and target.get("game_final") and r.get("actual_note"):
                    ui.label(r["actual_note"]).classes("text-xs text-gray-500")

            if dev.value:
                render_dev(r)

    def render_dev(r: dict) -> None:
        with ui.expansion("Developer details", value=True).classes("w-full").mark("dev-details"):
            mv = r.get("model_version") or {}
            for k, v in [("confidence", r.get("confidence")), ("projection_method", r.get("projection_method")),
                         ("in_training_window", r.get("in_training_window")),
                         ("games_this_season", r.get("games_this_season")),
                         ("model data_through", mv.get("data_through")), ("model fit", mv.get("fit_timestamp")),
                         ("team source", r["player"].get("team_source")), ("bounds_applied", r.get("bounds_applied")),
                         ("unpredicted", ", ".join(r.get("unpredicted") or []))]:
                ui.label(f"{k}: {v}").classes("text-xs font-mono")
            if r.get("volume_rate_detail"):
                ui.label("volume / rate detail:").classes("text-xs font-semibold mt-2")
                for k, v in r["volume_rate_detail"].items():
                    ui.label(f"  {k}: {v:.4f}").classes("text-xs font-mono")
            with ui.row().classes("items-end gap-2 mt-2"):
                season_in = ui.number("Season", value=(r.get("target") or {}).get("season"), format="%d") \
                    .classes("w-24").mark("override-season")
                week_in = ui.number("Week", value=(r.get("target") or {}).get("week"), format="%d") \
                    .classes("w-20").mark("override-week")
                ui.button("Project week", on_click=lambda: select(
                    selection["gsis_id"], int(season_in.value), int(week_in.value))).mark("override-go")
                ui.button("Next game", on_click=lambda: select(selection["gsis_id"])).props("flat")

    with card_area:
        render_card()
    dev.on_value_change(lambda _e: render_card.refresh())

    # ---- loading state -> ready -----------------------------------------------
    def check_ready() -> None:
        if STATE["error"]:
            loading.clear()
            with loading:
                ui.label(f"Failed to load data: {STATE['error']}").classes("text-red-600")
            ready_timer.cancel()
        elif STATE["index"] is not None:
            loading.set_visibility(False)
            main.set_visibility(True)
            info = STATE["model_info"] or {}
            dt = info.get("data_through") or {}
            footer.set_text(f"Model: data through {dt.get('season')} week {dt.get('week')} · "
                            f"fit {info.get('fit_timestamp')} · stat-vector projections"
                            if info else "Model: no artifacts found (run python -m pipeline.train retrain)")
            ready_timer.cancel()

    ready_timer = ui.timer(0.25, check_ready)
    check_ready()


if __name__ in {"__main__", "__mp_main__"}:
    import argparse
    parser = argparse.ArgumentParser(description="Audible IQ player projections (localhost)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args, _ = parser.parse_known_args()
    ui.run(title="Audible IQ", host="127.0.0.1", port=args.port, reload=False, show=False)
