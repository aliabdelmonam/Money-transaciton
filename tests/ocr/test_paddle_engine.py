"""PaddleOCR line logic that needs no model: box merging, choosing between the two
recognizers' readings, masked-name tokens. Run with ``python -m pytest`` from the root."""
import pytest

from paddle_engine import boxes, masked, recognize


def test_overlapping_boxes_of_one_line_merge():
    # "750" and "0 EGP": the second detection starts inside the last digit
    assert boxes.merge_overlapping([[218, 424, 486, 564], [425, 458, 664, 565]]) == [[218, 424, 664, 565]]


def test_label_and_value_on_one_row_stay_apart():
    assert len(boxes.merge_overlapping([[76, 1165, 227, 1201], [582, 1164, 803, 1199]])) == 2


@pytest.mark.parametrize("ar, en, expected", [
    # Arabic line: the Latin model's junk is ignored
    (("المرجع", 0.96), ("jll", 0.4), "المرجع"),
    # Latin line: the Latin model wins ...
    (("ahmed9o@instapay", 0.99), ("ahmed90@instapay", 0.98), "ahmed90@instapay"),
    # ... but word breaks the Arabic model kept beat the Latin model's dropped ones
    (("To Instapay", 0.957), ("ToInstapay", 0.999), "To Instapay"),
    (("MahmoudFA", 0.99), ("Mahmoud F A", 0.98), "Mahmoud F A"),
    # mixed line: Latin part from the Latin model, Arabic from the Arabic one
    (("Ling Expen  عيدية", 0.92), ("Living Expenses - ", 0.8), "Living Expenses - عيدية"),
    (("LnExpen عيدية", 0.91), ("Living Expenses - ", 0.8), "Living Expenses - عيدية"),
    # edge noise from icons is stripped
    ((") EGP", 0.88), (") EGP", 0.9), "EGP"),
])
def test_choose(ar, en, expected):
    assert recognize.choose(ar, en)[0] == expected


def test_mixed_line_kept_when_latin_reading_does_not_match():
    assert recognize.choose(("Note ملاحظة", 0.9), ("xq", 0.5))[0] == "Note ملاحظة"


@pytest.mark.parametrize("text, expected", [
    ("ahmedabdalrahman9o@instapay", "ahmedabdalrahman90@instapay"),
    ("muhammedd002@instapay", "muhammedd002@instapay"),
    ("john.doe@instapay", "john.doe@instapay"),
    ("احمد عبد الرحمن", "احمد عبدالرحمن"),
])
def test_fix_text(text, expected):
    assert recognize.fix_text(text) == expected


@pytest.mark.parametrize("text, rtl, single, expected", [
    ("|", False, True, "I"),
    ("o", False, True, "O"),
    ("AHMED H", False, False, "AHMED H"),
    ("محمد3", True, False, "محمد"),
    ("7", True, True, "ح"),   # an Arabic initial read as its digit look-alike
    ("x", True, True, ""),
])
def test_masked_name_tokens(text, rtl, single, expected):
    assert masked._token(text, rtl, single) == expected
