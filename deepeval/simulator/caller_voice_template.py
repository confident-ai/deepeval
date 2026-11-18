import textwrap
from typing import List

from deepeval.dataset import Persona


class CallerVoiceTemplate:
    @staticmethod
    def describe_caller_voice(
        persona: Persona,
        language: str,
        *,
        delivery: bool,
        asides: bool,
        background_speakers: List[str],
    ) -> str:
        """Ask for the parts of a call that a persona implies but does not spell out."""
        speakers = "\n".join(
            f"{index}. {description}"
            for index, description in enumerate(background_speakers, start=1)
        )
        wanted = []
        if delivery:
            wanted.append(
                "`delivery`: a voice direction for the actor playing this "
                "person, written as exactly four labelled sentences on one "
                "line: `Affect: ... Tone: ... Pacing: ... Emotion: ...`. "
                "Affect is who they sound like, including apparent age. Tone "
                "is the quality of the voice. Pacing is speed, rhythm and "
                "pauses. Emotion is what they feel and how openly it shows. "
                "Make each one vivid and physical (sighs, breaths, a shake "
                "in the voice, words hit hard) so it cannot be mistaken for "
                "a neutral voice. Work every one of the voice settings below "
                "into it. Describe only what a listener could hear: leave "
                "out what the person wants and facts about them."
            )
        if asides:
            wanted.append(
                f"`asides`: three short things, in {language}, this person would "
                "say to someone else in the room while on the phone. None of "
                "them is addressed to whoever they are calling."
            )
        if background_speakers:
            wanted.append(
                f"`background_lines`: for each background speaker below, in "
                f"order, a list of three short things in {language} they would "
                "say in the room while the call is going on."
            )
        requested = "\n".join(f"- {item}" for item in wanted)
        return textwrap.dedent(
            """\
            You are preparing a simulated phone call. Below is the persona of the caller.

            {persona}

            Voice settings:
            {voice}

            Background speakers:
            {speakers}

            Return a JSON object with these keys, and no others:
            {requested}

            JSON Output:
            """
        ).format(
            persona=persona.prompt_block(),
            voice=persona.voice_brief().replace("\n", "\n" + " " * 12),
            speakers=speakers or "None",
            requested=requested,
        )
