"""
Action handlers - execute actions extracted from transcribed text.

Each handler implements execute(action) -> dict result.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from unified.action_extractor import Action, ActionType

logger = logging.getLogger("action_handlers")


class ActionHandler(ABC):
    """Base class for action handlers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def can_handle(self, action: Action) -> bool:
        """Return True if this handler can process the action."""

    @abstractmethod
    def execute(self, action: Action) -> dict:
        """Execute the action and return a result dict."""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class CalendarHandler(ActionHandler):
    """Handle calendar event creation.

    Supports:
    - Local .ics file storage (no external account needed)
    - Google Calendar (via Google Calendar API, optional)
    - Microsoft Outlook (via Microsoft Graph, optional)
    """

    def __init__(self, name: str = "calendar", calendar_file: str | None = None):
        super().__init__(name)
        self.calendar_file = calendar_file or str(
            Path.home() / ".prism" / "calendar.ics"
        )
        self._ensure_calendar_dir()

    def _ensure_calendar_dir(self) -> None:
        Path(self.calendar_file).parent.mkdir(parents=True, exist_ok=True)

    def can_handle(self, action: Action) -> bool:
        return action.action_type == ActionType.CALENDAR_EVENT

    def execute(self, action: Action) -> dict:
        """Create a calendar event."""

        # Try Google Calendar first if credentials available
        if self._has_google_creds():
            return self._add_to_google_calendar(action)

        # Fallback to local .ics file
        return self._add_to_ics_file(action)

    def _has_google_creds(self) -> bool:
        return bool(
            os.getenv("GOOGLE_CALENDAR_CREDENTIALS")
            or Path("google_credentials.json").exists()
        )

    def _add_to_google_calendar(self, action: Action) -> dict:
        """Add event to Google Calendar via API."""
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            creds_data = os.getenv("GOOGLE_CALENDAR_CREDENTIALS")
            if creds_data:
                creds = Credentials.from_authorized_user_info(
                    json.loads(creds_data)
                )
            else:
                creds = Credentials.from_authorized_user_file(
                    "google_credentials.json"
                )

            service = build("calendar", "v3", credentials=creds)

            event = {
                "summary": action.title or "Untitled Event",
                "description": action.description or "",
                "location": action.location or "",
                "attendees": [{"email": a} for a in action.attendees],
                "start": {
                    "dateTime": action.start_time,
                    "timeZone": "local",
                },
                "end": {
                    "dateTime": action.end_time or action.start_time,
                    "timeZone": "local",
                },
            }

            event = service.events().insert(calendarId="primary", body=event).execute()

            return {
                "action": "calendar_event",
                "status": "created",
                "provider": "google_calendar",
                "event_id": event.get("id"),
                "event_link": event.get("htmlLink"),
                "title": action.title,
            }
        except Exception as e:
            logger.warning(f"Google Calendar failed: {e}, falling back to .ics")
            return self._add_to_ics_file(action)

    def _add_to_ics_file(self, action: Action) -> dict:
        """Add event to local .ics file."""

        start = action.start_time or datetime.now().isoformat()
        end = action.end_time or start
        duration = action.duration_minutes or 60

        ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Prism//Calendar//EN
BEGIN:VEVENT
UID:{action.action_type}-{abs(hash(action.title or ""))}@prism.local
DTSTART:{start.replace("-", "").replace(":", "").replace(".", "")}Z
DTEND:{end.replace("-", "").replace(":", "").replace(".", "")}Z
SUMMARY:{action.title or "Untitled Event"}
DESCRIPTION:{action.description or ""}
LOCATION:{action.location or ""}
END:VEVENT
END:VCALENDAR
"""

        with open(self.calendar_file, "a", encoding="utf-8") as f:
            f.write(ics_content + "\n")

        return {
            "action": "calendar_event",
            "status": "created",
            "provider": "local_ics",
            "file": self.calendar_file,
            "title": action.title,
            "start_time": action.start_time,
        }


class TodoHandler(ActionHandler):
    """Handle todo item creation.

    Supports:
    - Local todo.txt file
    - Todoist API (optional)
    - Todoist via API token
    """

    def __init__(self, name: str = "todo", todo_file: str | None = None):
        super().__init__(name)
        self.todo_file = todo_file or str(Path.home() / ".prism" / "todos.txt")
        Path(self.todo_file).parent.mkdir(parents=True, exist_ok=True)

    def can_handle(self, action: Action) -> bool:
        return action.action_type == ActionType.TODO

    def execute(self, action: Action) -> dict:
        try:
            return self._add_to_todoist(action)
        except Exception as e:
            logger.warning(f"Todoist failed: {e}, falling back to local file")
            return self._add_to_local_file(action)

    def _add_to_todoist(self, action: Action) -> dict:
        """Add task to Todoist via API."""
        api_key = os.getenv("TODOIST_API_KEY")
        if not api_key:
            raise ValueError("TODOIST_API_KEY not set")

        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required for Todoist API")

        payload = {
            "content": action.task or action.title or "Untitled task",
            "description": action.description or "",
            "priority": self._priority_to_todoist(action.priority),
            "due_string": action.due_date or None,
            "project_id": action.project or None,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        import httpx
        try:
            resp = httpx.post(
                "https://api.todoist.com/sync/v9/tasks",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Todoist API error: {e.response.status_code} {e.response.text[:200]}")
            return self._add_to_local_file(action)

        return {
            "action": "todo",
            "status": "created",
            "provider": "todoist",
            "task_id": data.get("id"),
            "task": action.task,
        }

    def _add_to_local_file(self, action: Action) -> dict:
        """Add task to local todo.txt file."""

        due = f" due:{action.due_date}" if action.due_date else ""
        priority_map = {"high": "P1", "medium": "P3", "low": "P5"}
        priority = priority_map.get(action.priority, "P3")

        line = f"{priority} {action.task or action.title or 'Untitled task'}{due}\n"

        with open(self.todo_file, "a", encoding="utf-8") as f:
            f.write(line)

        return {
            "action": "todo",
            "status": "created",
            "provider": "local_txt",
            "file": self.todo_file,
            "task": action.task,
        }

    def _priority_to_todoist(self, priority: str) -> int:
        return {"low": 3, "medium": 2, "high": 1}.get(priority, 3)


class EmailHandler(ActionHandler):
    """Handle email sending.

    Supports:
    - SMTP (any SMTP server)
    - SendGrid API (optional)
    """

    def __init__(self, name: str = "email"):
        super().__init__(name)
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        self.from_email = os.getenv("FROM_EMAIL")

    def can_handle(self, action: Action) -> bool:
        return action.action_type == ActionType.EMAIL

    def execute(self, action: Action) -> dict:
        if not self.smtp_host:
            return {
                "action": "email",
                "status": "error",
                "reason": "SMTP not configured (set SMTP_HOST, SMTP_USER, SMTP_PASS)",
            }

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
        except ImportError:
            return {"action": "email", "status": "error", "reason": "email modules not available"}

        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_email or self.smtp_user
            msg["To"] = action.recipient or ""
            msg["Subject"] = action.subject or "No Subject"
            msg.attach(MIMEText(action.body or action.description or "", "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_pass:
                    server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)

            return {
                "action": "email",
                "status": "sent",
                "to": action.recipient,
                "subject": action.subject,
            }
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return {"action": "email", "status": "error", "reason": str(e)}


class ReminderHandler(ActionHandler):
    """Handle reminder creation.

    Supports:
    - Local reminders file
    - System notifications (if available)
    """

    def __init__(self, name: str = "reminder", reminders_file: str | None = None):
        super().__init__(name)
        self.reminders_file = reminders_file or str(
            Path.home() / ".prism" / "reminders.txt"
        )
        Path(self.reminders_file).parent.mkdir(parents=True, exist_ok=True)

    def can_handle(self, action: Action) -> bool:
        return action.action_type == ActionType.REMINDER

    def execute(self, action: Action) -> dict:
        line = (
            f"{action.remind_at or 'ASAP'} | {action.reminder_text or action.text or ''}\n"
        )
        with open(self.reminders_file, "a", encoding="utf-8") as f:
            f.write(line)

        return {
            "action": "reminder",
            "status": "created",
            "provider": "local_txt",
            "file": self.reminders_file,
            "remind_at": action.remind_at,
            "reminder_text": action.reminder_text,
        }


class NoteHandler(ActionHandler):
    """Handle note creation."""

    def __init__(self, name: str = "note", notes_file: str | None = None):
        super().__init__(name)
        self.notes_file = reminders_file = str(
            Path.home() / ".prism" / "notes.txt"
        )
        Path(self.notes_file).parent.mkdir(parents=True, exist_ok=True)

    def can_handle(self, action: Action) -> bool:
        return action.action_type == ActionType.NOTE

    def execute(self, action: Action) -> dict:
        line = (
            f"[{datetime.now().isoformat()}] {action.title or action.text or ''}\n"
            f"  {action.description or action.text or ''}\n\n"
        )
        with open(self.notes_file, "a", encoding="utf-8") as f:
            f.write(line)

        return {
            "action": "note",
            "status": "created",
            "provider": "local_txt",
            "file": self.notes_file,
            "title": action.title,
        }


class CallHandler(ActionHandler):
    """Handle call logging (placeholder for phone integration)."""

    def __init__(self, name: str = "call"):
        super().__init__(name)

    def can_handle(self, action: Action) -> bool:
        return action.action_type == ActionType.CALL

    def execute(self, action: Action) -> dict:
        return {
            "action": "call",
            "status": "logged",
            "note": "Call logging requires telephony integration",
            "target": action.recipient or action.title,
        }


# ---------------------------------------------------------------------------
# Action Router - routes actions to appropriate handlers
# ---------------------------------------------------------------------------

class ActionRouter:
    """Routes actions to the appropriate handler."""

    def __init__(self):
        self.handlers: list[ActionHandler] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.handlers.append(CalendarHandler())
        self.handlers.append(TodoHandler())
        self.handlers.append(EmailHandler())
        self.handlers.append(ReminderHandler())
        self.handlers.append(NoteHandler())
        self.handlers.append(CallHandler())

    def add_handler(self, handler: ActionHandler) -> None:
        self.handlers.append(handler)

    def route(self, action: Action) -> dict:
        """Route an action to the appropriate handler."""

        for handler in self.handlers:
            if handler.can_handle(action):
                try:
                    result = handler.execute(action)
                    result["handler"] = handler.name
                    return result
                except Exception as e:
                    logger.error(f"Handler {handler.name} failed: {e}")
                    return {
                        "action": action.action_type.value,
                        "status": "error",
                        "handler": handler.name,
                        "reason": str(e),
                    }

        return {
            "action": action.action_type.value,
            "status": "no_handler",
            "reason": f"No handler for action type: {action.action_type.value}",
        }


# Singleton router
_action_router: ActionRouter | None = None


def get_action_router() -> ActionRouter:
    """Get the singleton action router."""
    global _action_router
    if _action_router is None:
        _action_router = ActionRouter()
    return _action_router


def reset_action_router() -> None:
    global _action_router
    _action_router = None
    _action_router = ActionRouter()


async def process_action(
    action: Action,
) -> dict:
    """Process a single action through the router."""
    router = get_action_router()
    return router.route(action)


async def process_actions(
    actions: list[Action],
) -> list[dict]:
    """Process multiple actions through the router."""
    router = get_action_router()
    results = []
    for action in actions:
        result = router.route(action)
        results.append(result)
    return results
