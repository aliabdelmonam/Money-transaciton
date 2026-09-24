"""Vocabulary for transaction_extractor.py.

Everything receipt-specific lives here, so supporting a new app, bank or layout is
usually just a matter of adding words below. Labels are compared after normalization
(case, punctuation, Arabic letter variants such as أ/إ/ا, ى/ي, ة/ه) and with fuzzy
tolerance for OCR typos, so each wording only needs to be listed once.
"""

# Field label -> where its value goes. "sender.*" / "receiver.*" fill a party;
# "*.account" values are classified by what they look like (phone, IPA handle, bank account).
FIELD_LABELS = {
    "amount": [
        "Amount", "Transfer Amount", "Total amount", "Amount Sent", "Transferred Amount", "Sent Amount",
        "المبلغ", "المبلغ المحول", "المبلغ الإجمالي المحول", "قيمة التحويل", "قيمة المعاملة", "المبلغ المرسل",
    ],
    "fees": [
        "Fees", "Fee", "Transaction Fee", "Transaction Fees", "Service Fee", "Service Fees", "Commission",
        "رسوم", "الرسوم", "رسوم المعاملة", "رسوم الخدمة", "رسوم التحويل", "العمولة",
    ],
    "total": [
        "Total", "Total Amount Deducted", "Total Deducted", "Amount Deducted", "Total Paid", "Grand Total",
        "المبلغ الكلي", "الإجمالي", "إجمالي المبلغ", "المبلغ المخصوم", "إجمالي المخصوم", "المبلغ المدفوع",
    ],
    "reference": [
        "Reference", "Reference Number", "Reference No", "Ref", "Ref No", "Transaction ID", "Transaction Number",
        "Transaction No", "Transaction Reference", "Operation ID", "Receipt Number",
        "المرجع", "الرقم المرجعي", "رقم المرجع", "رقم العملية", "رقم المعاملة", "رقم التحويل", "رقم الإيصال",
    ],
    "datetime": [
        "Date", "Time", "Date & Time", "Date and Time", "Transaction Date", "Transaction Time",
        "التاريخ", "الوقت", "التاريخ والوقت", "تاريخ العملية", "تاريخ المعاملة", "وقت العملية",
    ],
    "note": [
        "Note", "Notes", "Purpose", "Description", "Comment", "Remarks", "Reason",
        "ملاحظة", "ملاحظات", "الغرض", "الوصف", "سبب التحويل",
    ],
    "type": ["Transaction Type", "Service", "نوع العملية", "نوع المعاملة", "الخدمة"],
    "sender.name": ["Sender Name", "Payer Name", "From Name", "اسم المرسل", "اسم الراسل"],
    "sender.account": [
        "Sender Number", "Sender Account", "Sender Mobile", "From Account", "Wallet Number", "Your Wallet",
        "رقم المرسل", "حساب المرسل", "رقم المحفظة", "رقم محفظتك",
    ],
    "receiver.name": [
        "Receiver Name", "Recipient Name", "Beneficiary Name", "To Name",
        "اسم المستقبل", "اسم المستلم", "اسم المستفيد", "اسم المرسل إليه",
    ],
    "receiver.account": [
        "Receiver Number", "Recipient Number", "Receiver Mobile", "Recipient Mobile", "Receiver Account",
        "Beneficiary Account", "To Account",
        "رقم المرسل إليه", "رقم المستقبل", "رقم المستلم", "رقم المستفيد", "حساب المستفيد",
    ],
}

# Headers that open a party block ("From" followed by name / account lines).
PARTY_HEADERS = {
    "sender": ["From", "Sender", "Sent From", "Paid From", "من", "المرسل", "من حساب", "من محفظة"],
    "receiver": [
        "To", "Receiver", "Recipient", "Beneficiary", "Sent To", "Paid To",
        "إلى", "المستلم", "المستفيد", "المستقبل", "المرسل إليه",
    ],
}

# Text after a party header that names the channel rather than the person ("To Mobile Wallet").
CHANNELS = [
    "Mobile Wallet", "Wallet", "Bank Account", "Account", "Card", "InstaPay", "IPA", "Mobile Number",
    "محفظة", "محفظة إلكترونية", "حساب بنكي", "حساب", "بطاقة", "انستاباي", "رقم موبايل",
]

# Checked in this order ("unsuccessful" must hit "failed" before "success").
STATUS_KEYWORDS = {
    "failed": ["failed", "failure", "declined", "rejected", "unsuccessful", "فشل", "فشلت", "مرفوض", "مرفوضة", "لم تتم"],
    "pending": ["pending", "in progress", "under review", "قيد التنفيذ", "قيد المعالجة", "جاري التنفيذ"],
    "success": ["success", "successful", "successfully", "completed", "approved", "بنجاح", "ناجحة", "تمت", "تم"],
}

# Latin keywords of 4+ letters also match inside a word, since logos often glue a glyph on ("Daxis").
PROVIDERS = {
    "InstaPay": ["instapay", "انستاباي", "انستا باي"],
    "Axis": ["axis"],
    "Vodafone Cash": ["vodafone cash", "vodafone", "فودافون كاش", "فودافون"],
    "Orange Cash": ["orange cash", "orange money", "اورنج كاش"],
    "Etisalat Cash": ["etisalat cash", "e& cash", "e& money", "اتصالات كاش"],
    "WE Pay": ["we pay", "وي باي"],
    "Fawry": ["fawry", "myfawry"],
    "Meeza": ["meeza", "ميزة"],
    "Telda": ["telda"],
    "valU": ["valu"],
}

BANKS = [
    "CIB", "NBE", "QNB", "HSBC", "AAIB", "ADIB", "NBK", "SAIB", "FAB", "ENBD", "Emirates NBD", "Alex Bank",
    "Banque Misr", "Bank Misr", "Banque du Caire", "Arab African", "Credit Agricole", "Attijariwafa",
    "بنك مصر", "البنك الأهلي المصري", "البنك الأهلي", "بنك القاهرة", "البنك التجاري الدولي",
    "بنك الإسكندرية", "بنك قطر الوطني",
]

CURRENCIES = {
    "EGP": ["EGP", "LE", "L.E", "E£", "ج.م", "ج م", "جم", "جنيه مصري", "جنيه", "جنية"],
    "USD": ["USD", "US$", "$", "دولار"],
    "EUR": ["EUR", "€", "يورو"],
    "SAR": ["SAR", "ريال سعودي", "ريال"],
    "AED": ["AED", "درهم"],
}

# Known OCR misreads, applied to every line before anything else: (regex, replacement).
OCR_FIXES = [
    (r"(?<![A-Za-z])[Pρp]\.?\s?2(?=\s*\d|\s*$)", "ج.م "),  # Arabic "ج.م" read as "P.2" / "ρ2"
    (r"(?<=\d)\s*[I|l]\s*(?=\d{1,2}:\d{2})", " "),         # "2026 | 10:11" read as "2026I 10:11"
]
