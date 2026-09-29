"""One local IPL strategy agent that plans, calls tools, and grounds its answer."""

from __future__ import annotations

import json
import re
from typing import Any

from agentic.agent import AgentResponse
from agentic.analytics import GoldDataQueryTool
from agentic.tools import GoldAnalyticsTool, PredictionTool
from rag.settings import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


ALLOWED_TABLES = {
    "player_matchups", "player_form", "team_performance", "toss_analysis",
    "venue_strategy", "phase_analysis",
}


class IPLStrategyAgent:
    """Plan over local tools with Mistral; keep every numeric result tool-owned."""

    def __init__(
        self,
        query_tool: GoldDataQueryTool | None = None,
        rag_retriever=None,
        prediction_tool: PredictionTool | None = None,
        chat_model=None,
    ) -> None:
        self.query_tool = query_tool or GoldDataQueryTool()
        self.analytics_tool = GoldAnalyticsTool(self.query_tool)
        self.rag_retriever = rag_retriever
        self.prediction_tool = prediction_tool or PredictionTool()
        self.chat_model = chat_model

    def answer(self, question: str) -> AgentResponse:
        question = question.strip()
        response = AgentResponse(question, "")
        if not question:
            response.answer = "Please enter an IPL strategy question."
            response.limitations.append("No question was provided.")
            return response
        if re.search(r"\b(?:season|seasonal|year by year|by year|annual)\b", question.casefold()):
            response.interpretation = "The local Gold snapshots contain overall aggregates, not season-level rows."
            response.answer = "I cannot give a season-by-season answer from the current Gold snapshots; they aggregate the available IPL records across seasons."
            response.limitations.append("A season column or season-level Gold table is required for this question.")
            return response

        plan, planner_note = self._plan(question)
        selected_tables = list(dict.fromkeys(plan["tables"] + self._rule_tables(question)))
        selected_tables = self._validate_table_selection(question, selected_tables)
        if "player_matchups" in selected_tables:
            sides = re.split(r"\b(?:against|versus|vs\.?)\b", question, maxsplit=1, flags=re.I)
            if len(sides) != 2 or not (
                self.analytics_tool._resolve_batter(sides[0])
                and self.analytics_tool._resolve_bowler(sides[1])
            ):
                selected_tables.remove("player_matchups")
        response.interpretation = (
            f"{OLLAMA_CHAT_MODEL} selected tools for this question; deterministic Gold queries calculate structured statistics."
            if not planner_note
            else "Mistral planning was unavailable, so a conservative local intent router selected relevant tools."
        )
        if planner_note:
            response.limitations.append(planner_note)

        evidence: list[dict[str, Any]] = []
        if selected_tables:
            gold_evidence, notes = self.analytics_tool.run(question, selected_tables)
            evidence.extend(gold_evidence)
            response.limitations.extend(notes)
            response.tools_used.extend(f"Gold query: {table}" for table in dict.fromkeys(
                item["table"] for item in gold_evidence
            ))

        if plan["use_rag"] or self._should_retrieve(question):
            try:
                if self.rag_retriever is None:
                    self.rag_retriever = self._new_retriever()
                retriever = self.rag_retriever
                retrieved = retriever.search(question, top_k=4, sources=selected_tables or None)
                retrieved = self._filter_rag_results(question, selected_tables, retrieved)
                evidence.extend({"table": f"RAG:{item['source']}", "text": item["text"], "score": item.get("score")} for item in retrieved)
                response.tools_used.append("Qdrant RAG retriever")
                if not retrieved:
                    response.limitations.append("Qdrant returned no relevant Gold text documents.")
            except Exception as exc:
                response.limitations.append(f"RAG retrieval is unavailable: {exc}")

        wants_prediction = plan["use_prediction"] or self._should_predict(question)
        if wants_prediction:
            prediction, note = self.prediction_tool.run(question)
            response.tools_used.append("Chase prediction tool")
            if prediction:
                evidence.append({"table": "chase_prediction", **prediction})
            elif note:
                response.limitations.append(note)

        response.data_sources = list(dict.fromkeys(
            item["table"] for item in evidence if item.get("table")
        ))
        response.supporting_metrics = self._metrics(evidence)
        if not evidence:
            if wants_prediction and response.limitations:
                response.answer = response.limitations[-1]
            else:
                response.answer = (
                    "I could not find relevant Gold evidence for that question. Try asking about a player matchup or form, "
                    "team record, toss decision, venue chase rate, or innings phase."
                )
            return response

        answer_evidence = evidence + ([{"table": "tool_limitations", "notes": response.limitations}] if response.limitations else [])
        response.answer, generated_by_model, used_guard = self._generate_answer(question, answer_evidence)
        if generated_by_model:
            response.tools_used.append(f"Ollama answer generation: {OLLAMA_CHAT_MODEL}")
        else:
            response.tools_used.append("Evidence summary fallback")
        if used_guard:
            response.tools_used.append("Evidence-grain guard")
        return response

    def _validate_table_selection(self, question: str, tables: list[str]) -> list[str]:
        """Drop planner guesses that do not match available question entities/intents."""
        lower = question.casefold()
        approved = []
        for table in tables:
            if table == "player_form":
                named_player = self.analytics_tool._resolve_batter(question) is not None
                general_player_query = any(term in lower for term in (
                    "player", "batter", "top scorer", "top run", "compare", "form",
                ))
                if not (named_player or general_player_query):
                    continue
            if table == "team_performance":
                named_team = self.analytics_tool._resolve_team(question) is not None
                if not named_team and not any(term in lower for term in ("team", "franchise", "win percentage", "win rate")):
                    continue
            if table == "venue_strategy" and not any(term in lower for term in (
                "venue", "chase", "chasing", "bat first", "wankhede", "eden gardens",
            )):
                continue
            if table == "toss_analysis" and not any(term in lower for term in ("toss", "field or bat")):
                continue
            if table == "phase_analysis" and not any(term in lower for term in (
                "phase", "powerplay", "middle over", "death over", "run rate", "bowler", "bowling",
            )):
                continue
            approved.append(table)
        return approved

    def _plan(self, question: str) -> tuple[dict[str, Any], str | None]:
        fallback = {"tables": [], "use_rag": False, "use_prediction": False}
        try:
            from langchain_ollama import ChatOllama

            model = self.chat_model or ChatOllama(
                model=OLLAMA_CHAT_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0,
                format="json",
                num_predict=96,
                keep_alive="10m",
                client_kwargs={"timeout": 90.0},
            )
            prompt = (
                "You are the planner for one IPL strategy assistant. Choose only tools; do not answer the question. "
                "Return JSON with keys tables (array), use_rag (boolean), use_prediction (boolean). "
                f"Allowed Gold table names: {', '.join(sorted(ALLOWED_TABLES))}. "
                "Choose every table needed for numerical IPL facts. Use RAG for semantic/contextual retrieval. "
                "Use prediction only when a win-probability or decision prediction is requested. "
                "Do not invent table names. Question: " + question
            )
            message = model.invoke(prompt)
            raw = message.content if hasattr(message, "content") else str(message)
            parsed = json.loads(raw)
            tables = [name for name in parsed.get("tables", []) if name in ALLOWED_TABLES]
            return {
                "tables": tables,
                "use_rag": parsed.get("use_rag", False) is True,
                "use_prediction": parsed.get("use_prediction", False) is True,
            }, None
        except Exception as exc:
            return fallback, f"Mistral tool planning is unavailable ({type(exc).__name__}); local routing was used."

    def _new_retriever(self):
        from rag.retriever import GoldRAGRetriever
        return GoldRAGRetriever()

    def _filter_rag_results(self, question: str, selected_tables: list[str], hits: list[dict]) -> list[dict]:
        """Keep semantic matches aligned with any entity resolved by Gold tools."""
        exact_keys: dict[str, str] = {}
        if "team_performance" in selected_tables:
            team = self.analytics_tool._resolve_team(question)
            if team:
                exact_keys["team_performance"] = team.casefold()
        if "venue_strategy" in selected_tables:
            venues = self.query_tool.tables["venue_strategy"]["venue"].dropna().astype(str).unique()
            venue = self.analytics_tool._resolve_named_location(question, venues)
            if venue:
                exact_keys["venue_strategy"] = venue.casefold()
        if "player_form" in selected_tables:
            player = self.analytics_tool._resolve_batter(question)
            if player:
                exact_keys["player_form"] = player.casefold()
        if "phase_analysis" in selected_tables:
            phase_words = {"powerplay": "powerplay", "middle overs": "middle", "death overs": "death"}
            lower = question.casefold()
            phase = next((value for phrase, value in phase_words.items() if phrase in lower), None)
            if phase:
                exact_keys["phase_analysis"] = phase
        return [
            hit for hit in hits
            if float(hit.get("score", 1.0)) >= 0.45
            and (hit.get("source") not in exact_keys or hit.get("key", "").casefold() == exact_keys[hit["source"]])
        ]

    def _generate_answer(self, question: str, evidence: list[dict[str, Any]]) -> tuple[str, bool, bool]:
        prompt_evidence = json.dumps(evidence, ensure_ascii=False, default=str)
        try:
            from langchain_ollama import ChatOllama

            model = self.chat_model or ChatOllama(
                model=OLLAMA_CHAT_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0,
                num_predict=64,
                keep_alive="10m",
                client_kwargs={"timeout": 90.0},
            )
            message = model.invoke([
                ("system", (
                    "You are an IPL analyst. Answer only from the supplied tool evidence. "
                    "Never invent or calculate statistics. You may compare provided values directly, but do not derive new rates. "
                    "Do not infer head-to-head or chase-specific performance from separate overall team and venue aggregates. "
                    "Cite factual claims with their table name in square brackets. Separate historical trends from predictions. "
                    "If evidence is incomplete, state the limitation. Answer in at most three concise sentences."
                )),
                ("human", f"Question: {question}\nTool evidence (JSON): {prompt_evidence}"),
            ])
            answer = message.content if hasattr(message, "content") else str(message)
            if answer.strip():
                safe_answer = self._guarded_answer(question, evidence)
                if safe_answer:
                    return safe_answer, True, True
                return answer.strip(), True, False
        except Exception:
            pass
        safe_answer = self._guarded_answer(question, evidence)
        if safe_answer:
            return safe_answer, False, True
        return self._evidence_summary(evidence), False, False

    @staticmethod
    def _evidence_summary(evidence: list[dict[str, Any]]) -> str:
        lines = []
        for item in evidence:
            source = item.get("table", "evidence")
            if "rows" in item:
                lines.append(f"{source}: {json.dumps(item['rows'], ensure_ascii=False, default=str)}")
            elif "text" in item:
                lines.append(f"{source}: {item['text']}")
            elif source == "chase_prediction":
                lines.append(f"Model estimate: {item.get('win_probability')} win probability (chase model).")
        return "\n\n".join(lines)

    @staticmethod
    def _guarded_answer(question: str, evidence: list[dict[str, Any]]) -> str | None:
        """State evidence at its actual grain when the question exceeds the data."""
        lower = question.casefold()
        notes = [note for item in evidence if item.get("table") == "tool_limitations" for note in item.get("notes", [])]
        note_text = " ".join(notes).casefold()
        wants_decision = any(phrase in lower for phrase in ("should we", "recommend", "what should")) or bool(
            re.search(r"\bshould\b.*\b(?:chase|bat|field|win)\b", lower)
        )
        if "bowler-level death-over" in note_text:
            return (
                "The available `phase_analysis` table gives totals by innings phase, not by bowler. "
                "It cannot identify the most effective death-over bowlers; that requires bowler-by-over data."
            )
        if "team_performance contains win/loss outcomes only" in note_text:
            team = next((item for item in evidence if item.get("table") == "team_performance" and item.get("rows")), None)
            if team:
                row = team["rows"][0]
                return (
                    f"{row['team']}'s available team-level result is {row['wins']} wins from "
                    f"{row['matches_played']} matches ({row['win_percentage']}%) [team_performance]. "
                    "This table does not separate batting, bowling, phase, or current-roster strengths."
                )
        if not wants_decision or any(item.get("table") == "chase_prediction" for item in evidence):
            return None
        venue = next((item for item in evidence if item.get("table") == "venue_strategy" and item.get("rows")), None)
        team = next((item for item in evidence if item.get("table") == "team_performance" and item.get("rows")), None)
        model_missing = any("no fitted model artifact" in note.casefold() for note in notes)
        if venue and team:
            venue_row, team_row = venue["rows"][0], team["rows"][0]
            return (
                f"At {venue_row['venue']}, teams chasing won {venue_row['chasing_win_percentage']}% of "
                f"{venue_row['matches']} matches [venue_strategy]. {team_row['team']} won "
                f"{team_row['wins']} of {team_row['matches_played']} matches overall "
                f"({team_row['win_percentage']}%) [team_performance]. These separate historical aggregates "
                "do not establish this team's chase record at that venue. "
                + ("No chase prediction was produced because the model artifact is unavailable." if model_missing else "")
            )
        if venue:
            row = venue["rows"][0]
            return (
                f"At {row['venue']}, teams chasing won {row['chasing_win_percentage']}% of "
                f"{row['matches']} matches [venue_strategy]. This historical venue rate alone is not an "
                "individual match prediction. "
                + ("No chase prediction was produced because the model artifact is unavailable." if model_missing else "")
            )
        return "The available Gold aggregates do not provide enough information for an individualized chase recommendation."

    @staticmethod
    def _metrics(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        metrics = []
        for item in evidence:
            source = item.get("table", "evidence")
            if item.get("rows"):
                for row in item["rows"][:2]:
                    for key, value in row.items():
                        if isinstance(value, (int, float)) and len(metrics) < 12:
                            metrics.append({"label": f"{source} · {key.replace('_', ' ')}", "value": f"{value:g}"})
            elif source == "chase_prediction" and "win_probability" in item:
                metrics.append({"label": "Model win probability", "value": f"{item['win_probability']:.1%}"})
        return metrics

    @staticmethod
    def _should_retrieve(question: str) -> bool:
        lower = question.casefold()
        return any(term in lower for term in (
            "summarize", "summary", "context", "historical", "strength", "strategy", "why", "recommend", "should we",
        ))

    @staticmethod
    def _should_predict(question: str) -> bool:
        lower = question.casefold()
        return any(term in lower for term in (
            "predict", "probability", "win chance", "chance of winning", "should we chase", "should we bat",
        )) or bool(re.search(r"\bshould\b.*\b(?:chase|bat|field)\b|\bwill\b.*\bwin\b", lower))

    def _rule_tables(self, question: str) -> list[str]:
        lower = question.casefold()
        tables: list[str] = []
        matchup_sides = re.split(r"\b(?:against|versus|vs\.?)\b", question, maxsplit=1, flags=re.I)
        if len(matchup_sides) == 2:
            if self.analytics_tool._resolve_batter(matchup_sides[0]) and self.analytics_tool._resolve_bowler(matchup_sides[1]):
                tables.append("player_matchups")
        elif any(word in lower for word in ("batter", "player", "runs", "form", "compare")):
            if any(_contains_name(question, str(name)) for name in self.query_tool.tables["player_form"]["batter"].dropna().astype(str).unique()) or any(alias in lower for alias in ("kohli", "rohit sharma", "dhawan")):
                tables.append("player_form")
        if any(word in lower for word in ("team", "franchise", "strength", "csk", "rcb", "mi", "kkr", "srh", "pbks", "chennai super kings")):
            tables.append("team_performance")
        if any(word in lower for word in ("venue", "chase", "chasing", "bat first", "wankhede", "eden gardens")):
            tables.append("venue_strategy")
        if "toss" in lower or "field or bat" in lower:
            tables.append("toss_analysis")
        if any(word in lower for word in ("phase", "powerplay", "middle overs", "death overs", "run rate", "bowlers", "bowler")):
            tables.append("phase_analysis")
        return tables


def _contains_name(question: str, name: str) -> bool:
    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return f" {normalize(name)} " in f" {normalize(question)} "
