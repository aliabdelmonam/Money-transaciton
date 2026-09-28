"""Provider knowledge: labels, sentence patterns, banks. Add new wording here."""

# field -> label variants (Arabic + English)
GENERIC_LABELS = {
    "amount": ["amount", "transfer amount", "transferred amount", "amount sent",
               "المبلغ", "المبلغ المحول", "المبلغ الإجمالي المحول", "قيمة التحويل", "قيمة المعاملة"],
    "fees": ["fees", "fee", "transaction fees", "service fees",
             "رسوم", "الرسوم", "رسوم المعاملة", "رسوم التحويل", "رسوم الخدمة"],
    "total": ["total", "total amount", "total amount deducted", "total deducted",
              "المبلغ الكلي", "الإجمالي", "المبلغ الإجمالي", "إجمالي المبلغ", "إجمالي المبلغ المخصوم"],
    "reference": ["reference", "reference number", "reference no", "ref", "ref no",
                  "transaction id", "transaction number", "transaction ref",
                  "المرجع", "الرقم المرجعي", "رقم المرجع", "رقم العملية", "رقم المعاملة"],
    "date": ["date", "date time", "transaction time", "transaction date", "time",
             "التاريخ", "تاريخ العملية", "التاريخ والوقت", "الوقت"],
    "note": ["note", "notes", "description", "purpose", "ملاحظة", "ملاحظات", "الغرض"],
    "sender.name": ["sender name", "from name", "اسم المرسل", "اسم المحول"],
    "sender.phone": ["sender number", "sender mobile", "wallet number", "رقم المرسل", "رقم المحفظة"],
    "receiver.name": ["receiver name", "beneficiary name", "recipient name",
                      "اسم المستقبل", "اسم المستفيد", "اسم المرسل إليه"],
    "receiver.phone": ["receiver number", "receiver mobile", "recipient number", "beneficiary number",
                       "رقم المستقبل", "رقم المستفيد", "رقم المرسل إليه"],
}

# detect: keywords that identify the provider (matched on normalized text)
# label_overrides: label -> field, for wording that means something different in this app
# sentences: extra regexes with named groups amount / sender_phone / receiver_phone
PROVIDERS = {
    "instapay": {"detect": ["@instapay", "instapay", "انستاباي", "powered by", "ipn"],
                 "label_overrides": {}, "sentences": []},
    "axis": {"detect": ["axis", "الدفع تم بنجاح"],
             "label_overrides": {"رقم المحفظة": "sender.phone", "رقم المرسل إليه": "receiver.phone"},
             "sentences": []},
    # guessed name - screenshot with "Total Amount Deducted"
    "vodafone_cash": {"detect": ["total amount deducted", "payment has been processed"],
                      "label_overrides": {"total amount": "amount"}, "sentences": []},
    "wallet_simple": {"detect": ["you transferred", "you have transferred"],
                      "label_overrides": {}, "sentences": []},
}

_NUM = r"(?P<amount>\d[\d,]*(?:\.\d+)?)"
_PH_R = r"(?P<receiver_phone>\+?\d[\d ]{8,14}\d)"
_PH_S = r"(?P<sender_phone>\+?\d[\d ]{8,14}\d)"
GENERIC_SENTENCES = [
    r"you\s+(?:have\s+)?(?:successfully\s+)?(?:transferred|sent|paid)\s+(?:egp\s*)?" + _NUM
    + r"\s*(?:egp|le)?\.?\s+to\s+" + _PH_R,
    r"you\s+(?:have\s+)?received\s+(?:egp\s*)?" + _NUM + r"\s*(?:egp|le)?\.?\s+from\s+" + _PH_S,
    r"تم\s+(?:تحويل|ارسال)\s+(?:مبلغ\s+)?" + _NUM + r".{0,25}?(?:الي|الى)\s*" + _PH_R,
    r"تم\s+استلام\s+(?:مبلغ\s+)?" + _NUM + r".{0,25}?من\s*" + _PH_S,
]

ANCHOR_WORDS = {"sender": {"from", "من"}, "receiver": {"to", "الي"}}   # normalized forms

BANKS = {"banque misr": "Banque Misr", "بنك مصر": "Banque Misr",
         "national bank of egypt": "National Bank of Egypt", "nbe": "National Bank of Egypt",
         "البنك الاهلي": "National Bank of Egypt", "cib": "CIB", "البنك التجاري الدولي": "CIB",
         "qnb": "QNB", "banque du caire": "Banque du Caire", "بنك القاهره": "Banque du Caire",
         "alexbank": "AlexBank", "aaib": "AAIB", "hsbc": "HSBC", "adib": "ADIB",
         "telda": "Telda", "fawry": "Fawry"}

NOT_NAMES = ["powered by", "egp", "instapay", "انستاباي", "mobile wallet", "wallet",
             "محفظة", "محفظة موبايل", "success", "successful", "transaction successful",
             "بنجاح", "bank", "ج م", "le"]

CURRENCY_MARKERS = (" egp ", " ج م ", "جنيه", " le ")
SUCCESS_WORDS = ["success", "بنجاح", "تمت العمليه", "تم الدفع"]
FAIL_WORDS = ["unsuccessful", "failed", "declined", "فشل", "مرفوض", "لم تتم"]
