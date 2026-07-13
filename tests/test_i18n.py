"""Task-local language selection must be safe for concurrent Web sessions."""

from contextvars import copy_context

from utility_safety_ai.i18n import _, set_language


def test_language_selection_is_context_local():
    set_language("en")
    chinese_context = copy_context()
    chinese_context.run(set_language, "zh-hans")

    assert _("person") == "person"
    assert chinese_context.run(_, "person") == "人员"
