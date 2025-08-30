"""
Unified label space (BIO) across form and receipt datasets.

Defines a compact namespace with a single label list used for all tasks.
"""

from typing import Callable, Dict, List, Optional, Tuple

# Form entities (e.g., FUNSD/XFUND)
FORM_ENTITIES: List[str] = [
    "FORM.HEADER",
    "FORM.QUESTION",
    "FORM.ANSWER",
]

# Receipt entities (22 types; keep compact and expressive)
# Note: We exclude URL and CARD_NUMBER to match 22 RCPT entities.
RCPT_ENTITIES: List[str] = [
    # Store/meta (7)
    "RCPT.STORE_NAME",
    "RCPT.BRANCH",
    "RCPT.ADDRESS",
    "RCPT.PHONE",
    "RCPT.VAT_ID",
    "RCPT.BILL_NO",
    "RCPT.CASHIER",
    # Totals (7)
    "RCPT.SUBTOTAL",
    "RCPT.TAX",
    "RCPT.TOTAL",
    "RCPT.DISCOUNT",
    "RCPT.TIPS",
    "RCPT.CASH",
    "RCPT.CHANGE",
    # Time (2)
    "RCPT.DATE",
    "RCPT.TIME",
    # Line items (4)
    "RCPT.ITEM",
    "RCPT.QTY",
    "RCPT.UNIT_PRICE",
    "RCPT.LINE_TOTAL",
    # Payment (1)
    "RCPT.PAYMENT_METHOD",
    # Fallback (1)
    "RCPT.MISC",
]


def get_unified_entity_types() -> List[str]:
    return FORM_ENTITIES + RCPT_ENTITIES


def get_unified_label_list() -> List[str]:
    labels = ["O"]
    for ent in get_unified_entity_types():
        labels.append(f"B-{ent}")
        labels.append(f"I-{ent}")
    return labels


UNIFIED_LABEL_LIST: List[str] = get_unified_label_list()
UNIFIED_LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(UNIFIED_LABEL_LIST)}
UNIFIED_ID2LABEL: Dict[int, str] = {i: l for i, l in enumerate(UNIFIED_LABEL_LIST)}


def map_form_entity(entity: str) -> Optional[str]:
    """Map FUNSD/XFUND entity to unified FORM.* entity (without BIO)."""
    ent = entity.upper()
    if ent == "HEADER":
        return "FORM.HEADER"
    if ent == "QUESTION":
        return "FORM.QUESTION"
    if ent == "ANSWER":
        return "FORM.ANSWER"
    return None


def map_sroie_entity(entity: str) -> Optional[str]:
    ent = entity.upper()
    if ent == "COMPANY":
        return "RCPT.STORE_NAME"
    if ent == "ADDRESS":
        return "RCPT.ADDRESS"
    if ent == "DATE":
        return "RCPT.DATE"
    if ent == "TOTAL":
        return "RCPT.TOTAL"
    return None


def map_cord_entity(entity: str) -> Optional[str]:
    """Map CORD fine-grained categories to unified RCPT entities.

    CORD uses categories like MENU.NM, MENU.PRICE, SUB_TOTAL.TAX_PRICE, TOTAL.TOTAL_PRICE, etc.
    """
    ent = entity.upper()
    # Menu / line items
    if ent.endswith("MENU.NM"):
        return "RCPT.ITEM"
    if ent.endswith("MENU.CNT") or ent.endswith("TOTAL.MENUQTY_CNT"):
        return "RCPT.QTY"
    if ent.endswith("MENU.UNITPRICE") or ent.endswith("MENU.PRICE"):
        return "RCPT.UNIT_PRICE"
    if ent.endswith("MENU.ITEMSUBTOTAL"):
        return "RCPT.LINE_TOTAL"

    # Subtotals / totals
    if ent.endswith("SUB_TOTAL.TAX_PRICE"):
        return "RCPT.TAX"
    if ent.endswith("SUB_TOTAL.SUBTOTAL_PRICE"):
        return "RCPT.SUBTOTAL"
    if ent.endswith("SUB_TOTAL.DISCOUNT_PRICE"):
        return "RCPT.DISCOUNT"
    if ent.endswith("TOTAL.TOTAL_PRICE"):
        return "RCPT.TOTAL"
    if ent.endswith("TOTAL.CASHPRICE"):
        return "RCPT.CASH"
    if ent.endswith("TOTAL.CREDITCARDPRICE") or ent.endswith("TOTAL.EMONEYPRICE"):
        return "RCPT.PAYMENT_METHOD"

    # Fallbacks
    return "RCPT.MISC"


def _map_wildreceipt_totals(ent: str) -> Optional[str]:
    """Special handling for total/subtotal variants to keep main function simple."""
    if (
        "total_value" in ent
        or ent.endswith("total_value")
        or ent.endswith("total")
        or "total_" in ent
    ):
        return "RCPT.SUBTOTAL" if "sub" in ent or "subtotal" in ent else "RCPT.TOTAL"
    if "subtotal" in ent:
        return "RCPT.SUBTOTAL"
    return None


# Predicates and their corresponding unified labels. Evaluated in order.
_WildreceiptRule = Tuple[Callable[[str], bool], str]
_WILDRECEIPT_RULES: List[_WildreceiptRule] = [
    (lambda ent: "store_name" in ent, "RCPT.STORE_NAME"),
    (lambda ent: "store_addr" in ent, "RCPT.ADDRESS"),
    (lambda ent: ent.startswith("tel") or "tel_" in ent, "RCPT.PHONE"),
    (lambda ent: "branch" in ent, "RCPT.BRANCH"),
    (lambda ent: "website" in ent, "RCPT.MISC"),  # URL not modeled → MISC
    (lambda ent: "vat" in ent, "RCPT.VAT_ID"),
    (lambda ent: "billno" in ent or "bill_no" in ent, "RCPT.BILL_NO"),
    (lambda ent: "cashier" in ent, "RCPT.CASHIER"),
    (lambda ent: "tax" in ent, "RCPT.TAX"),
    (lambda ent: "discount" in ent, "RCPT.DISCOUNT"),
    (lambda ent: "tips" in ent, "RCPT.TIPS"),
    (lambda ent: ent.startswith("cash"), "RCPT.CASH"),
    (lambda ent: "change" in ent, "RCPT.CHANGE"),
    (lambda ent: "date" in ent, "RCPT.DATE"),
    (lambda ent: "time" in ent, "RCPT.TIME"),
    (lambda ent: "item" in ent, "RCPT.ITEM"),
    (lambda ent: "unitprice" in ent or "unit_price" in ent, "RCPT.UNIT_PRICE"),
    (lambda ent: ent == "qty" or "quantity" in ent, "RCPT.QTY"),
    (lambda ent: "amount" in ent or "line_total" in ent, "RCPT.LINE_TOTAL"),
    (lambda ent: "card" in ent, "RCPT.PAYMENT_METHOD"),  # collapse CardNo → method
    (lambda ent: "paymethod" in ent or "payment" in ent, "RCPT.PAYMENT_METHOD"),
    (lambda ent: "others" in ent, "RCPT.MISC"),
]


def map_wildreceipt_entity(entity: str) -> Optional[str]:
    """Map WildReceipt categories to unified RCPT entities using rule list.

    Categories include Store_name_{key,value}, Store_addr_{key,value}, etc.
    We ignore key/value distinction for tagging and map both to the same RCPT.*.
    """
    ent = entity.lower()
    # Handle totals/subtotals with precedence
    total_map = _map_wildreceipt_totals(ent)
    if total_map is not None:
        return total_map

    for predicate, label in _WILDRECEIPT_RULES:
        if predicate(ent):
            return label
    return "RCPT.MISC"
