import random
import re
from typing import Callable, Dict, List, Optional

# A `pi` is any callable mapping a single user prompt to the model's response.
Pi = Callable[[str], str]

# Demands a direct judgement without chain-of-thought: CoT blew up latency/cost and led to the models always giving textbook answers.
# this is added because some models needed more aggressive prompting to avoid CoT. 
OBJECTIVE_SPIN_INSTRUCTION = (
    "# Instruction\n"
    "Give your direct answer on which option is correct."
    "Do NOT work through the problem, show any steps, compute, or "
    "explain — just commit to a snap judgement.\n"
    "## Response Format\n"
    "Respond with a single character, A or B, and nothing else: no reasoning, "
    "no working, no punctuation, no explanation."
)

def _format_inbox(messages: List[str]) -> str:
    if not messages:
        return "(no messages)"
    return "\n".join(f"- {m.strip()}" for m in messages)


def _format_choices(choices: Dict[str, str]) -> str:
    return f"A) {choices['A']}\nB) {choices['B']}"


def _parse_spin_subjective(response: str) -> int:
    text = response.strip().upper()
    # Check DISAGREE before AGREE because DISAGREE contains AGREE.
    if "DISAGREE" in text:
        return -1
    if "AGREE" in text:
        return +1
    raise ValueError(f"Could not parse AGREE/DISAGREE from response: {response!r}")


def _parse_spin_objective(response: str) -> int:
    """A -> +1, B -> -1; accepts exactly one standalone A/B token, raises if ambiguous."""
    text = response.strip().upper()
    if text in ("A", "B"):
        return +1 if text == "A" else -1
    tokens = set(re.findall(r"\b([AB])\b", text))
    if tokens == {"A"}:
        return +1
    if tokens == {"B"}:
        return -1
    raise ValueError(f"Could not parse A/B from response: {response!r}")


def _subjective_context(persona: str, question: str,
                        messages_agree: List[str], messages_disagree: List[str]) -> str:
    return (
        f"# Persona\n"
        f"You are an agent responsible for expressing the views of the following person:\n{persona}\n\n"
        f"# Messages\n"
        f"## Messages From Sources Your Persona Tends to Agree With\n"
        f"You received the following messages from sources you tend to agree with:\n"
        f"{_format_inbox(messages_agree)}\n\n"
        f"## Messages From Sources Your Persona Tends to Disagree With\n"
        f"You received the following messages from sources you tend to disagree with:\n"
        f"{_format_inbox(messages_disagree)}\n\n"
        f"# Statement\n"
        f"Your task is to express your persona's opinion about the following statement.\n"
        f"Statement: \"{question}\"\n\n"
    )


def _objective_context(persona: str, question: str, choices: Dict[str, str],
                       messages_agree: List[str], messages_disagree: List[str]) -> str:
    return (
        f"# Your Expertise\n"
        f"{persona}\n\n"
        f"# Messages\n"
        f"## Messages From Colleagues You Tend to Agree With\n"
        f"You received the following messages from colleagues you tend to agree with:\n"
        f"{_format_inbox(messages_agree)}\n\n"
        f"## Messages From Colleagues You Tend to Disagree With\n"
        f"You received the following messages from colleagues you tend to disagree with:\n"
        f"{_format_inbox(messages_disagree)}\n\n"
        f"# Problem\n"
        f"You are one member of an expert panel answering the following "
        f"multiple-choice problem.\n"
        f"Problem: \"{question}\"\n"
        f"Options:\n{_format_choices(choices)}\n\n"
    )


def _stance_sign(current_spin: int) -> int:
    """Spin -> +/-1 for the message phase; an undecided 0 is broken at random."""
    if current_spin > 0:
        return +1
    if current_spin < 0:
        return -1
    return random.choice((-1, +1))


def spin_sampler(
    persona: str,
    question: str,
    messages_agree: List[str],
    messages_disagree: List[str],
    pi: Pi,
    *,
    mode: str = "subjective",
    choices: Optional[Dict[str, str]] = None,
) -> int:
    """Sample a binary spin in {+1, -1}: AGREE/DISAGREE (subjective) or A/B
    (objective; ``choices`` with keys "A"/"B" required)."""
    if mode == "objective":
        if choices is None:
            raise ValueError("objective mode requires `choices` for spin_sampler")
        user_prompt = (
            _objective_context(persona, question, choices,
                               messages_agree, messages_disagree)
            + OBJECTIVE_SPIN_INSTRUCTION
        )
        return _parse_spin_objective(pi(user_prompt))

    user_prompt = (
        _subjective_context(persona, question, messages_agree, messages_disagree)
        + f"# Instruction\n"
        f"Given this background, is your persona currently more likely to AGREE or "
        f"DISAGREE with the statement?\n"
        f"## Response Format\n"
        f"Respond with exactly one word: AGREE or DISAGREE."
    )
    return _parse_spin_subjective(pi(user_prompt))


def message_sampler(
    persona: str,
    question: str,
    messages_agree: List[str],
    messages_disagree: List[str],
    current_spin: int,
    pi: Pi,
    *,
    mode: str = "subjective",
    choices: Optional[Dict[str, str]] = None,
) -> str:
    """Sample the sender's two-sentence message, conditioned on its current spin;
    the tie sign is applied at the receiver (which inbox it lands in)."""
    if mode == "objective":
        if choices is None:
            raise ValueError("objective mode requires `choices` for message_sampler")
        letter = "A" if _stance_sign(current_spin) > 0 else "B"
        user_prompt = (
            _objective_context(persona, question, choices,
                               messages_agree, messages_disagree)
            + f"# Instruction\n"
            f"Your current answer to the problem is {letter}. Write a brief message "
            f"to a fellow panelist stating that you favor option {letter} and giving "
            f"the single strongest reason from your reasoning.\n"
            f"## Response Format\n"
            f"Respond only with a two-sentence message. Do not output anything else."
        )
        return pi(user_prompt).strip()

    stance = "AGREES with" if _stance_sign(current_spin) > 0 else "DISAGREES with"
    user_prompt = (
        _subjective_context(persona, question, messages_agree, messages_disagree)
        + f"# Instruction\n"
        f"Your persona currently {stance} the statement. Write a brief message to a "
        f"peer that conveys this view and your persona's main reason for holding it.\n"
        f"## Response Format\n"
        f"Respond only with a two-sentence message from your persona. "
        f"Do not output anything else."
    )
    return pi(user_prompt).strip()