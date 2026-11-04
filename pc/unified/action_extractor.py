"""
Action extraction from transcribed text.

Takes raw transcribed text and uses an AI provider to extract actionable
intents: calendar events, todo items, emails, reminders, etc.

Returns structured Action objects that can be executed by various
handlers (calendar, todo, email, etc.).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from unified.ai_client import AIClient, AIResponse

logger = logging.getLogger("action_extractor")


class ActionType(Enum):
    CALENDAR_EVENT = "calendar_event"
    TODO = "todo"
    EMAIL = "email"
    REMINDER = "reminder"
    NOTE = "note"
    CALL = "call"
    UNKNOWN = "unknown"


@dataclass
class Action:
    """A single actionable item extracted from transcribed text."""

    action_type: ActionType
    confidence: float  # 0.0 to 1.0
    raw_text: str  # original text segment
    normalized_text: str  # cleaned/normalized text

    # Calendar fields
    title: str | None = None
    description: str | None = None
    start_time: str | None = None  # ISO 8601
    end_time: str | None = None  # ISO 8601
    duration_minutes: int | None = None
    location: str | None = None
    attendees: list[str] = field(default_factory=list)

    # Todo fields
    task: str | None = None
    due_date: str | None = None  # ISO 8601 date
    priority: str = "medium"  # low, medium, high
    project: str | None = None

    # Email fields
    recipient: str | None = None
    subject: str | None = None
    body: str | None = None

    # Reminder fields
    reminder_text: str | None = None
    remind_at: str | None = None  # ISO 8601

    # Generic
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type.value,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "title": self.title,
            "description": self.description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_minutes": self.duration_minutes,
            "location": self.location,
            "attendees": self.attendees,
            "task": self.task,
            "due_date": self.due_date,
            "priority": self.priority,
            "project": self.project,
            "recipient": self.recipient,
            "subject": self.subject,
            "body": self.body,
            "reminder_text": self.reminder_text,
            "remind_at": self.remind_at,
            "metadata": self.metadata,
        }


class ActionExtractor:
    """Extracts actionable items from transcribed text using an AI provider."""

    SYSTEM_PROMPT = """You are an assistant that extracts actionable items from
transcribed speech. Given a transcript, identify any actionable items and
return them as a JSON array.

Each action should have:
- action_type: one of "calendar_event", "todo", "email", "reminder", "note", "call", "unknown"
- confidence: 0.0 to 1.0
- raw_text: the original text segment
- normalized_text: cleaned version of the text
- title: for events, the event title
- description: description/notes
- start_time: ISO 8601 datetime if a specific time is mentioned
- end_time: ISO 8601 datetime if end time is mentioned
- duration_minutes: duration in minutes if mentioned
- location: location if mentioned
- attendees: list of people mentioned
- task: for todos, the task description
- due_date: ISO 8601 date if a due date is mentioned
- priority: low, medium, high
- project: project name if mentioned
- recipient: email recipient
- subject: email subject
- body: email body
- reminder_text: what to remind about
- remind_at: ISO 8601 when to remind

If no actionable items are found, return an empty array [].
Only return valid JSON, no markdown formatting, no explanation.

Examples:
Input: "Remind me to call mom tomorrow at 3pm"
Output: [{"action_type": "reminder", "confidence": 0.9, "raw_text": "call mom tomorrow at 3pm", "normalized_text": "call mom tomorrow at 3pm", "reminder_text": "call mom", "remind_at": "2024-01-15T15:00:00"}]

Input: "Schedule a meeting with Sarah tomorrow at 2pm for 30 minutes to discuss the project"
Output: [{"action_type": "calendar_event", "confidence": 0.95, "raw_text": "meeting with Sarah tomorrow at 2pm for 30 minutes", "normalized_text": "meeting with Sarah tomorrow at 2pm for 30 minutes", "title": "Meeting with Sarah", "start_time": "2024-01-15T14:00:00", "end_time": "2024-01-15T14:30:00", "duration_minutes": 30, "attendees": ["Sarah"], "description": "discuss the project"}]

Input: "Buy milk and eggs"
Output: [{"action_type": "todo", "confidence": 0.8, "raw_text": "Buy milk and eggs", "normalized_text": "Buy milk and eggs", "task": "Buy milk and eggs", "priority": "medium"}]
"""

    def __init__(self, client: AIClient | None = None):
        self.client = client or AIClient()

    async def extract(
        self,
        text: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[Action]:
        """Extract actionable items from transcribed text."""

        if not text or not text.strip():
            return []

        provider_obj = self.client.get_provider(provider)
        if provider_obj is None:
            logger.warning("No AI provider available for action extraction")
            return []

        prompt = self._build_prompt(text)

        try:
            response = await self.client.complete(
                prompt=prompt,
                provider=provider,
                model=model,
                temperature=0.3,  # Low temperature for deterministic extraction
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"AI provider failed: {e}")
            return []

        return self._parse_response(response.text, text)

    def _build_prompt(self, text: str) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        return f"""Today is {today}. Tomorrow is {tomorrow}.

Transcribed text: "{text}"

Extract actionable items as JSON array. Return ONLY the JSON array, no markdown."""

    def _parse_response(self, response_text: str, original_text: str) -> list[Action]:
        """Parse the AI response into Action objects."""

        # Clean up the response - remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```"):
            # Remove markdown code block
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            # Try to extract JSON array from within the text
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        actions: list[Action] = []
        for item in data:
            try:
                action = self._parse_action(item, original_text)
                if action is not None:
                    actions.append(action)
            except Exception as e:
                logger.warning(f"Failed to parse action item: {e}")
                continue

        return actions

    def _parse_action(self, item: dict, original_text: str) -> Action | None:
        """Parse a single action item from parsed JSON."""

        action_type_str = item.get("action_type", "unknown").lower()
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.UNKNOWN

        confidence = float(item.get("confidence", 0.5))

        return Action(
            action_type=action_type,
            confidence=confidence,
            raw_text=item.get("raw_text", original_text),
            normalized_text=item.get("normalized_text", item.get("raw_text", original_text)),
            title=item.get("title"),
            description=item.get("description"),
            start_time=item.get("start_time"),
            end_time=item.get("end_time"),
            duration_minutes=item.get("duration_minutes"),
            location=item.get("location"),
            attendees=item.get("attendees", []),
            task=item.get("task"),
            due_date=item.get("due_date"),
            priority=item.get("priority", "medium"),
            project=item.get("project"),
            recipient=item.get("recipient"),
            subject=item.get("subject"),
            body=item.get("body"),
            reminder_text=item.get("reminder_text"),
            remind_at=item.get("remind_at"),
            metadata=item.get("metadata", {}),
        )


# Convenience function for quick extraction
async def extract_actions(
    text: str,
    provider: str | None = None,
    model: str | None = None,
) -> list[Action]:
    """Extract actionable items from text using the default AI client."""
    extractor = ActionExtractor()
    return await extractor.extract(text, provider=provider, model=model)


# Synchronous wrapper for non-async contexts
def extract_actions_sync(
    text: str,
    provider: str | None = None,
    model: str | None = None,
) -> list[Action]:
    """Synchronous wrapper for extract_actions."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(extract_actions(text, provider=provider, model=model))
