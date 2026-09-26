"""
Canonical AI-opinion prompt, shared byte-for-byte across Weibo and Twitter.

Single prompt, single model — this removes the need for the cross-lingual bias
correction the old two-model (GPT for Twitter, Kimi for Weibo) setup required.
"""

SYSTEM_PROMPT = """You will read a piece of text posted by a user on social media. The text may be written
in Chinese, English, or any other language — analyze it regardless of the language it is
written in. Please analyze the text's opinion toward AI technology and assign it an
opinion label. Your output must be a JSON object containing only one field, "opinion",
whose value must be one of: -2, -1, 0, 1, 2, or "cannot tell".

Label definitions:
2 = The text expresses a strongly positive attitude toward AI technology, clearly
asserting that the benefits of AI outweigh the harms and that AI brings significant
positive impacts to society (e.g., improving convenience, enhancing health and medical
services, creating economic opportunities, increasing learning or work efficiency,
improving safety, or supporting research and innovation), or expresses reliance on or
trust in AI.
1 = The text is overall positive toward AI, but the attitude is mild or includes
reservations (e.g., expresses support but without strong enthusiasm; believes AI is
"generally beneficial" while also mentioning risks or limitations; expresses
expectation, interest, or positive impressions, but not strong praise).
0 = The text is neutral or difficult to judge (e.g., mentions both pros and cons but
without a clear leaning).
-1 = The text is overall negative toward AI, but the attitude is mild or includes
reservations (e.g., expresses concern or opposition but does not fully reject AI;
believes AI is "risky or harmful" but acknowledges certain benefits; expresses caution,
unease, or negative views, but without strong condemnation).
-2 = The text expresses a strongly negative attitude toward AI technology, clearly
asserting that the harms outweigh the benefits and that AI brings significant negative
impacts to society (e.g., leading to unemployment, contributing to economic bubbles,
increasing privacy risks, reinforcing bias against marginalized groups, generating
misinformation or rumors, or posing safety threats), or expresses resistance to or
distrust in AI.
"cannot tell" = The text does not express any attitude toward AI (e.g., content is
unrelated to AI, or does not reflect a clear viewpoint).

Please return only a JSON object, for example:
{"opinion": 1}"""


def build_user_message(text: str) -> str:
    return f"Post text: {text}"
