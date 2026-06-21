import csv
import os
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import ActiveLoop, SlotSet
from rasa_sdk.executor import CollectingDispatcher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA = 0.5
BETA = 0.97
GAMMA = 0.999

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "user_data")

_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
_ENABLE_GROQ_RESPONSE = (
    os.environ.get("ENABLE_GROQ_RESPONSE", "false").lower() == "true"
)


def _groq_nlg(prompt: str, fallback: str) -> str:
    if not (_ENABLE_GROQ_RESPONSE and _GROQ_KEY):
        return fallback
    try:
        from groq import Groq

        client = Groq(api_key=_GROQ_KEY)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "És um assistente da Universidade de Aveiro. Responde sempre em português, de forma concisa e natural. Não inventes dados — usa apenas os fornecidos.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return fallback


INTENT_LABELS = {
    "ask_schedule": "consultar o teu horário",
    "ask_nearest_canteen": "saber a cantina mais próxima",
    "ask_menu": "ver a ementa de uma cantina",
    "ask_exam_info": "informações sobre um exame",
    "save_exam_date": "guardar a data de um exame",
}

INTENT_TO_FORM = {
    "ask_schedule": "schedule_form",
    "ask_nearest_canteen": "canteen_form",
    "ask_menu": "menu_form",
    "ask_exam_info": "exam_query_form",
    "save_exam_date": "save_exam_form",
}

DEFAULT_CANTEEN = "Refeitório de Santiago"


def load_cantinas() -> Dict[str, str]:
    rows = read_csv("cantinas.csv")
    return {r["local"].strip().lower(): r["cantina_mais_proxima"].strip() for r in rows}


# Map Python weekday() (0=Monday) to Portuguese day names used in the CSV
WEEKDAY_TO_PT = {
    0: "segunda",
    1: "terça",
    2: "quarta",
    3: "quinta",
    4: "sexta",
    5: "sábado",
    6: "domingo",
}

# Map common user-provided day strings to CSV day names
DAY_ALIASES: Dict[str, str] = {
    "segunda": "segunda",
    "segunda-feira": "segunda",
    "terça": "terça",
    "terca": "terça",
    "terça-feira": "terça",
    "terca-feira": "terça",
    "quarta": "quarta",
    "quarta-feira": "quarta",
    "quinta": "quinta",
    "quinta-feira": "quinta",
    "sexta": "sexta",
    "sexta-feira": "sexta",
    "sábado": "sábado",
    "sabado": "sábado",
    "domingo": "domingo",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    return (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )


def get_confidence(tracker: Tracker) -> float:
    """Return confidence of the latest recognised intent."""
    try:
        return float(tracker.latest_message.get("intent", {}).get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def read_csv(filename: str) -> List[Dict[str, str]]:
    """Read a CSV file from DATA_DIR and return a list of row dicts.

    Returns an empty list if the file is not found or cannot be parsed.
    """
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [row for row in reader]
    except FileNotFoundError:
        return []
    except Exception:
        return []


def resolve_query_date(date_str: str) -> str:
    """Convert 'hoje'/'amanhã'/etc to YYYY-MM-DD. Returns today if unrecognised."""
    if not date_str:
        return datetime.today().strftime("%Y-%m-%d")
    d = date_str.strip().lower()
    if d in ("hoje", "today"):
        return datetime.today().strftime("%Y-%m-%d")
    if d in ("amanhã", "amanha", "tomorrow"):
        return (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    return datetime.today().strftime("%Y-%m-%d")


def resolve_day(day_str: str) -> str:
    """Convert a user-supplied day string to a CSV-compatible Portuguese weekday.

    Handles: hoje, amanhã, and weekday names (with aliases).
    Returns the resolved day string in lowercase, or the original string if
    no resolution is possible.
    """
    if not day_str:
        return ""

    normalised = day_str.strip().lower()

    if normalised in ("hoje", "today"):
        return WEEKDAY_TO_PT.get(datetime.today().weekday(), normalised)

    if normalised in ("amanhã", "amanha", "tomorrow"):
        tomorrow = datetime.today() + timedelta(days=1)
        return WEEKDAY_TO_PT.get(tomorrow.weekday(), normalised)

    return DAY_ALIASES.get(normalised, normalised)


# ---------------------------------------------------------------------------
# Action: action_pre_schedule  (confidence gate — runs BEFORE schedule_form)
# ---------------------------------------------------------------------------


class ActionPreSchedule(Action):
    def name(self) -> Text:
        return "action_pre_schedule"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        confidence = get_confidence(tracker)

        if confidence < ALPHA:
            dispatcher.utter_message(response="utter_reject")
            return [SlotSet("original_confidence", None)]

        if confidence < BETA:
            # Explicit confirm: wait for affirm/deny, action_confirm_save_exam dispatches
            dispatcher.utter_message(response="utter_explicit_confirm_schedule")
            return [
                SlotSet("pending_intent", "ask_schedule"),
                SlotSet("original_confidence", confidence),
            ]

        # BETA+ → start form directly
        return [
            ActiveLoop("schedule_form"),
            SlotSet("original_confidence", confidence),
        ]


# ---------------------------------------------------------------------------
# Action: action_pre_canteen
# ---------------------------------------------------------------------------

class ActionPreCanteen(Action):
    def name(self) -> Text:
        return "action_pre_canteen"

    def run(self, dispatcher, tracker, domain):
        confidence = get_confidence(tracker)
        if confidence < ALPHA:
            dispatcher.utter_message(response="utter_reject")
            return [SlotSet("original_confidence", None)]
        if confidence < BETA:
            dispatcher.utter_message(response="utter_explicit_confirm_canteen")
            return [SlotSet("pending_intent", "ask_nearest_canteen"), SlotSet("original_confidence", confidence)]
        return [ActiveLoop("canteen_form"), SlotSet("original_confidence", confidence)]


# ---------------------------------------------------------------------------
# Action: action_pre_menu
# ---------------------------------------------------------------------------

class ActionPreMenu(Action):
    def name(self) -> Text:
        return "action_pre_menu"

    def run(self, dispatcher, tracker, domain):
        confidence = get_confidence(tracker)
        if confidence < ALPHA:
            dispatcher.utter_message(response="utter_reject")
            return [SlotSet("original_confidence", None)]
        if confidence < BETA:
            dispatcher.utter_message(response="utter_explicit_confirm_menu")
            return [SlotSet("pending_intent", "ask_menu"), SlotSet("original_confidence", confidence)]
        return [ActiveLoop("menu_form"), SlotSet("original_confidence", confidence)]


# ---------------------------------------------------------------------------
# Action: action_pre_exam
# ---------------------------------------------------------------------------

class ActionPreExam(Action):
    def name(self) -> Text:
        return "action_pre_exam"

    def run(self, dispatcher, tracker, domain):
        confidence = get_confidence(tracker)
        if confidence < ALPHA:
            dispatcher.utter_message(response="utter_reject")
            return [SlotSet("original_confidence", None)]
        if confidence < BETA:
            dispatcher.utter_message(response="utter_explicit_confirm_exam")
            return [SlotSet("pending_intent", "ask_exam_info"), SlotSet("original_confidence", confidence)]
        return [ActiveLoop("exam_query_form"), SlotSet("original_confidence", confidence)]


# ---------------------------------------------------------------------------
# Action: action_pre_save_exam
# ---------------------------------------------------------------------------

class ActionPreSaveExam(Action):
    def name(self) -> Text:
        return "action_pre_save_exam"

    def run(self, dispatcher, tracker, domain):
        confidence = get_confidence(tracker)
        if confidence < ALPHA:
            dispatcher.utter_message(response="utter_reject")
            return [SlotSet("original_confidence", None)]
        if confidence < BETA:
            dispatcher.utter_message(response="utter_explicit_confirm_exam")
            return [SlotSet("pending_intent", "save_exam_date"), SlotSet("original_confidence", confidence)]
        return [ActiveLoop("save_exam_form"), SlotSet("original_confidence", confidence)]


# ---------------------------------------------------------------------------
# Action: action_get_schedule
# ---------------------------------------------------------------------------


class ActionGetSchedule(Action):
    def name(self) -> Text:
        return "action_get_schedule"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Use confidence stored by action_pre_schedule (captured on original intent)
        stored = tracker.get_slot("original_confidence")
        confidence = stored if stored is not None else 1.0

        day_slot = tracker.get_slot("day")
        resolved = resolve_day(day_slot)
        time_of_day = tracker.get_slot("time_of_day")

        rows = read_csv("horario.csv")

        # Filter by day (normalize both sides to handle accent variants)
        filtered = [
            r for r in rows if normalize(r.get("dia", "")) == normalize(resolved)
        ]

        # Filter by time_of_day if provided
        if time_of_day and filtered:
            tod = time_of_day.strip().lower()
            if tod in ("manhã", "manha", "morning"):
                filtered = [
                    r for r in filtered if r.get("hora_inicio", "00:00") < "13:00"
                ]
            elif tod in ("tarde", "afternoon", "evening"):
                filtered = [
                    r for r in filtered if r.get("hora_inicio", "00:00") >= "13:00"
                ]

        # Filter by exact time if provided
        time_slot = tracker.get_slot("time")
        if time_slot and filtered:
            time_str = time_slot.strip().lower()
            time_str = time_str.replace("às ", "").replace("h", "").strip()
            if ":" not in time_str:
                time_str = time_str + ":00"
            filtered = [
                r
                for r in filtered
                if r.get("hora_inicio", "") <= time_str < r.get("hora_fim", "")
            ]

        SCHEDULE_RESETS = [
            SlotSet("day", None),
            SlotSet("time", None),
            SlotSet("time_of_day", None),
        ]

        if not filtered:
            dispatcher.utter_message(text="Não tens aulas nesse período.")
            return SCHEDULE_RESETS

        # Build reply with implicit confirm prefix when in BETA..GAMMA zone
        prefix = ""
        if BETA <= confidence < GAMMA:
            prefix = f"Para {resolved}, "

        sorted_rows = sorted(filtered, key=lambda x: x.get("hora_inicio", ""))
        lines = [f"{prefix}tens as seguintes aulas:"]
        for r in sorted_rows:
            hora_inicio = r.get("hora_inicio", "")
            hora_fim = r.get("hora_fim", "")
            disciplina = r.get("disciplina", "?")
            sala = r.get("sala", "?")
            lines.append(f"  - {disciplina}: {hora_inicio}–{hora_fim}, sala {sala}")
        fallback = "\n".join(lines)

        groq_prompt = (
            f"O utilizador perguntou pelo horário de {resolved}. "
            f"Dados: {sorted_rows}. "
            f"Gera uma resposta natural e concisa em português. Inclui disciplina, hora e sala."
        )
        dispatcher.utter_message(text=_groq_nlg(groq_prompt, fallback))
        return SCHEDULE_RESETS


# ---------------------------------------------------------------------------
# Action: action_get_nearest_canteen
# ---------------------------------------------------------------------------


class ActionGetNearestCanteen(Action):
    def name(self) -> Text:
        return "action_get_nearest_canteen"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        confidence = get_confidence(tracker)

        if confidence < ALPHA:
            dispatcher.utter_message(response="utter_reject")
            return []

        if confidence < BETA:
            dispatcher.utter_message(response="utter_explicit_confirm_canteen")
            return [SlotSet("pending_intent", "ask_nearest_canteen")]

        location = tracker.get_slot("location")
        location_lower = location.strip().lower()

        cantinas = load_cantinas()
        canteen = None
        for key, name in sorted(cantinas.items(), key=lambda x: -len(x[0])):
            if key in location_lower:
                canteen = name
                break

        if canteen is None:
            dispatcher.utter_message(text=f"Não conheço '{location}'.")
            return []

        # dispatcher.utter_message(text="Hm-hm.")

        if BETA <= confidence < GAMMA:
            message = f"Perto de {location}, a cantina mais próxima é {canteen}."
        else:
            message = f"A cantina mais próxima de {location} é {canteen}."

        dispatcher.utter_message(text=message)
        return []


# ---------------------------------------------------------------------------
# Action: action_get_known_locations
# ---------------------------------------------------------------------------


class ActionGetKnownLocations(Action):
    def name(self) -> Text:
        return "action_get_known_locations"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        cantinas = load_cantinas()
        by_canteen: Dict[str, list] = {}
        for local, canteen in cantinas.items():
            by_canteen.setdefault(canteen, []).append(local)

        lines = ["Conheço os seguintes locais do campus:"]
        for _, locais in sorted(by_canteen.items()):
            for loc in sorted(locais):
                lines.append(f"  - {loc}")

        dispatcher.utter_message(text="\n".join(lines))
        return []


# ---------------------------------------------------------------------------
# Action: action_get_menu
# ---------------------------------------------------------------------------


class ActionGetMenu(Action):
    def name(self) -> Text:
        return "action_get_menu"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        confidence = get_confidence(tracker)

        if confidence < ALPHA:
            dispatcher.utter_message(response="utter_reject")
            return []

        if confidence < BETA:
            dispatcher.utter_message(response="utter_explicit_confirm_menu")
            return [SlotSet("pending_intent", "ask_menu")]

        canteen = tracker.get_slot("canteen")
        meal_type = tracker.get_slot("meal_type")
        query_date = tracker.get_slot("query_date")
        target_date = resolve_query_date(query_date)
        date_label = (
            "amanhã"
            if query_date and normalize(query_date) in ("amanha", "tomorrow")
            else "hoje"
        )
        rows = read_csv("ementa.csv")

        canteen_norm = normalize(canteen)
        meal_norm = normalize(meal_type)

        filtered = [
            r
            for r in rows
            if r.get("data", "").strip() == target_date
            and canteen_norm in normalize(r.get("cantina", ""))
            and meal_norm in normalize(r.get("tipo_refeicao", ""))
        ]

        MENU_RESETS = [
            SlotSet("canteen", None),
            SlotSet("meal_type", None),
            SlotSet("query_date", None),
        ]

        if not filtered:
            dispatcher.utter_message(
                text=f"Não há ementa disponível para {canteen} {date_label}."
            )
            return MENU_RESETS

        prefix = ""
        if BETA <= confidence < GAMMA:
            prefix = f"Para {canteen} ({meal_type}), "

        lines = [f"{prefix}a ementa de {date_label} é:"]
        for r in filtered:
            prato = r.get("prato", "?")
            vegetariano = r.get("vegetariano", "nao").strip().lower()
            veg_tag = " (V)" if vegetariano in ("sim", "s", "yes", "true", "1") else ""
            lines.append(f"  - {prato}{veg_tag}")

        dispatcher.utter_message(text="\n".join(lines))
        return MENU_RESETS


# ---------------------------------------------------------------------------
# Action: action_get_exam_info
# ---------------------------------------------------------------------------


class ActionGetExamInfo(Action):
    def name(self) -> Text:
        return "action_get_exam_info"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        confidence = get_confidence(tracker)

        if confidence < ALPHA:
            dispatcher.utter_message(response="utter_reject")
            return []

        if confidence < BETA:
            dispatcher.utter_message(response="utter_explicit_confirm_exam")
            return [SlotSet("pending_intent", "ask_exam_info")]

        subject = tracker.get_slot("subject")
        subject_lower = subject.strip().lower()
        rows = read_csv("exames.csv")

        # Case-insensitive substring match on disciplina
        filtered = [
            r for r in rows if subject_lower in r.get("disciplina", "").strip().lower()
        ]

        EXAM_RESETS = [SlotSet("subject", None)]

        if not filtered:
            dispatcher.utter_message(text=f"Não encontrei exame para '{subject}'.")
            return EXAM_RESETS

        today_str = datetime.today().strftime("%Y-%m-%d")

        filtered.sort(key=lambda r: r.get("data", ""))

        # Prefer upcoming exams; fall back to all matches if none are upcoming
        upcoming = [r for r in filtered if r.get("data", "") >= today_str]
        display = upcoming if upcoming else filtered

        prefix = ""
        if BETA <= confidence < GAMMA:
            prefix = f"Sobre {subject}: "

        lines = []
        for r in display:
            disciplina = r.get("disciplina", subject)
            data = r.get("data") or "?"
            hora = r.get("hora") or "hora desconhecida"
            sala = r.get("sala") or "sala desconhecida"
            obs = r.get("observacoes", "").strip()

            line = f"Exame de {disciplina}: {data} às {hora} na sala {sala}."
            if obs:
                line += f" {obs}"
            lines.append(line)
        fallback = prefix + "\n".join(lines)

        groq_prompt = (
            f"O utilizador perguntou por informação sobre o exame de {subject}. "
            f"Dados: {display}. "
            f"Gera uma resposta natural e concisa em português. Inclui disciplina, data, hora e sala."
        )
        dispatcher.utter_message(text=_groq_nlg(groq_prompt, fallback))
        return EXAM_RESETS


# ---------------------------------------------------------------------------
# Action: action_save_exam_date
# ---------------------------------------------------------------------------


class ActionSaveExamDate(Action):
    def name(self) -> Text:
        return "action_save_exam_date"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        confidence = get_confidence(tracker)

        if confidence < ALPHA:
            dispatcher.utter_message(response="utter_reject")
            return []

        if confidence < BETA:
            dispatcher.utter_message(response="utter_explicit_confirm_exam")
            return [SlotSet("pending_intent", "save_exam_date")]

        subject = tracker.get_slot("subject")
        exam_date = tracker.get_slot("exam_date")
        exam_room = tracker.get_slot("exam_room") or ""

        raw_time = tracker.get_slot("exam_time") or tracker.get_slot("time") or ""
        # Normalize "9h" / "9:00" / "09h" → "09:00"
        t = raw_time.strip().lower().replace("h", ":")
        if t.endswith(":"):
            t = t[:-1]
        if ":" not in t and t.isdigit():
            t = t + ":00"
        parts = t.split(":")
        try:
            exam_time = (
                f"{int(parts[0]):02d}:{parts[1].zfill(2)}"
                if len(parts) == 2
                else raw_time
            )
        except (ValueError, IndexError):
            exam_time = raw_time

        if not subject:
            dispatcher.utter_message(response="utter_ask_exam_subject_save")
            return []

        if not exam_date:
            dispatcher.utter_message(response="utter_ask_exam_date")
            return []

        # Normalize date: "21 de junho" → "2026-06-21", "amanhã" → tomorrow
        import re

        MONTH_MAP = {
            "janeiro": "01",
            "fevereiro": "02",
            "março": "03",
            "abril": "04",
            "maio": "05",
            "junho": "06",
            "julho": "07",
            "agosto": "08",
            "setembro": "09",
            "outubro": "10",
            "novembro": "11",
            "dezembro": "12",
        }

        date_str = exam_date.strip().lower()
        normalized_date = None

        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            normalized_date = date_str
        elif "amanhã" in date_str or "amanha" in date_str:
            normalized_date = (datetime.today() + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
        elif "hoje" in date_str:
            normalized_date = datetime.today().strftime("%Y-%m-%d")
        else:
            for month_name, month_num in MONTH_MAP.items():
                if month_name in date_str:
                    day_match = re.search(r"(\d{1,2})", date_str)
                    if day_match:
                        day = day_match.group(1).zfill(2)
                        year = "2026"
                        normalized_date = f"{year}-{month_num}-{day}"
                        break

        if not normalized_date:
            normalized_date = exam_date  # keep as-is if can't parse

        today_str = datetime.today().strftime("%Y-%m-%d")
        is_past = normalized_date < today_str

        # Check if an exam already exists for this subject
        rows = read_csv("exames.csv")
        existing = [
            r for r in rows if subject.lower() in r.get("disciplina", "").lower()
        ]

        SAVE_RESETS = [
            SlotSet("subject", None),
            SlotSet("exam_date", None),
            SlotSet("exam_time", None),
            SlotSet("exam_room", None),
            SlotSet("time", None),
        ]

        # Store pending data in slots for later confirmation if needed
        pending_slots = SAVE_RESETS + [
            SlotSet("pending_exam_subject", subject),
            SlotSet("pending_exam_date", normalized_date),
            SlotSet("pending_exam_time", exam_time),
            SlotSet("pending_exam_room", exam_room),
        ]

        if is_past:
            dispatcher.utter_message(
                text=f"Atenção: {normalized_date} já passou. Queres mesmo guardar este exame?"
            )
            return pending_slots

        if existing:
            old = existing[0]
            dispatcher.utter_message(
                text=(
                    f"Já existe um exame de {subject} registado para "
                    f"{old.get('data')} às {old.get('hora')}. Queres actualizar?"
                )
            )
            return pending_slots

        # No confirmation needed — save directly
        return self._write_exam(
            subject,
            normalized_date,
            exam_time,
            exam_room,
            dispatcher,
            rows,
            SAVE_RESETS,
        )

    def _write_exam(
        self,
        subject: str,
        date_val: str,
        time_val: str,
        room_val: str,
        dispatcher: CollectingDispatcher,
        rows: List[Dict[str, str]],
        resets: List[Dict[Text, Any]] = None,
    ) -> List[Dict[Text, Any]]:
        path = os.path.join(DATA_DIR, "exames.csv")
        rows = [
            r for r in rows if subject.lower() not in r.get("disciplina", "").lower()
        ]
        rows.append(
            {
                "disciplina": subject,
                "data": date_val,
                "hora": time_val or "",
                "sala": room_val or "",
                "observacoes": "",
            }
        )
        rows.sort(key=lambda r: r.get("data", ""))
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["disciplina", "data", "hora", "sala", "observacoes"]
                )
                writer.writeheader()
                writer.writerows(rows)
            time_part = f" às {time_val}" if time_val else ""
            room_part = f" na sala {room_val}" if room_val else ""
            dispatcher.utter_message(
                text=f"Exame de {subject} guardado para {date_val}{time_part}{room_part}."
            )
        except Exception as e:
            dispatcher.utter_message(text=f"Erro ao guardar: {e}")
        return resets or []


# ---------------------------------------------------------------------------
# Action: action_confirm_save_exam
# ---------------------------------------------------------------------------


class ActionConfirmSaveExam(Action):
    def name(self) -> Text:
        return "action_confirm_save_exam"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Explicit-confirm tier: pending_intent set by pre-check actions
        pending_intent = tracker.get_slot("pending_intent")
        if pending_intent:
            form = INTENT_TO_FORM.get(pending_intent)
            dispatcher.utter_message(response="utter_affirm_continue")
            events = [
                SlotSet("pending_intent", None),
                SlotSet("original_confidence", None),
            ]
            if form:
                events.append(ActiveLoop(form))
            return events

        subject = tracker.get_slot("pending_exam_subject")
        date_val = tracker.get_slot("pending_exam_date")
        time_val = tracker.get_slot("pending_exam_time") or ""
        room_val = tracker.get_slot("pending_exam_room") or ""

        if not subject or not date_val:
            dispatcher.utter_message(text="Não há exame pendente para guardar.")
            return []

        rows = read_csv("exames.csv")
        rows = [
            r for r in rows if subject.lower() not in r.get("disciplina", "").lower()
        ]
        rows.append(
            {
                "disciplina": subject,
                "data": date_val,
                "hora": time_val,
                "sala": room_val,
                "observacoes": "",
            }
        )
        rows.sort(key=lambda r: r.get("data", ""))

        path = os.path.join(DATA_DIR, "exames.csv")
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["disciplina", "data", "hora", "sala", "observacoes"]
                )
                writer.writeheader()
                writer.writerows(rows)
            time_part = f" às {time_val}" if time_val else ""
            room_part = f" na sala {room_val}" if room_val else ""
            dispatcher.utter_message(
                text=f"Exame de {subject} guardado para {date_val}{time_part}{room_part}."
            )
        except Exception as e:
            dispatcher.utter_message(text=f"Erro ao guardar: {e}")

        return [
            SlotSet("pending_exam_subject", None),
            SlotSet("pending_exam_date", None),
            SlotSet("pending_exam_time", None),
            SlotSet("pending_exam_room", None),
        ]


# ---------------------------------------------------------------------------
# Action: action_handle_deny
# ---------------------------------------------------------------------------


class ActionHandleDeny(Action):
    def name(self) -> Text:
        return "action_handle_deny"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        pending_intent = tracker.get_slot("pending_intent")
        if pending_intent:
            dispatcher.utter_message(text="Ok, cancelado.")
            return [SlotSet("pending_intent", None)]

        dispatcher.utter_message(response="utter_exam_save_cancelled")
        return [
            SlotSet("pending_exam_subject", None),
            SlotSet("pending_exam_date", None),
            SlotSet("pending_exam_time", None),
            SlotSet("pending_exam_room", None),
        ]


# ---------------------------------------------------------------------------
# CONFREQ slot-ask actions — echo filled context while requesting next slot
# ---------------------------------------------------------------------------


class ActionAskTimeOfDay(Action):
    def name(self) -> Text:
        return "action_ask_time_of_day"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        day = tracker.get_slot("day")
        if day:
            dispatcher.utter_message(text=f"E de manhã ou de tarde na {day}?")
        else:
            dispatcher.utter_message(response="utter_ask_time_of_day")
        return []


class ActionAskMealType(Action):
    def name(self) -> Text:
        return "action_ask_meal_type"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        canteen = tracker.get_slot("canteen")
        if canteen:
            dispatcher.utter_message(text=f"E almoço ou jantar no {canteen}?")
        else:
            dispatcher.utter_message(response="utter_ask_meal_type")
        return []
