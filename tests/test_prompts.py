from prompts import SYSTEM_PROMPT, build_user_message


def test_build_user_message_format():
    assert build_user_message("hello world") == "Post text: hello world"


def test_build_user_message_no_platform_or_language_wording():
    message = build_user_message("some text")
    assert "Twitter" not in message
    assert "Weibo" not in message


def test_system_prompt_has_all_opinion_labels():
    for label in ["2 =", "1 =", "0 =", "-1 =", "-2 =", '"cannot tell" =']:
        assert label in SYSTEM_PROMPT


def test_system_prompt_requires_json_with_opinion_field():
    assert '"opinion"' in SYSTEM_PROMPT
    assert "JSON object" in SYSTEM_PROMPT


def test_system_prompt_is_language_and_platform_neutral():
    assert "Twitter" not in SYSTEM_PROMPT
    assert "Weibo" not in SYSTEM_PROMPT
    assert "regardless of the language" in SYSTEM_PROMPT
