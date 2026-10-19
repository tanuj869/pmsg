"""
State-name canonicalization for the PM Surya Ghar panel.

Every source file spells state names slightly differently
(e.g. "Jammu & Kashmir" vs "Jammu and Kashmir", "Telengana" vs "Telangana").
This module maps every variant encountered so far onto one canonical list,
so joins across datasets don't silently drop rows.

Import this in every notebook that loads a raw file:
    from state_mapping import normalize_state, CANONICAL_STATES
"""

CANONICAL_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Ladakh", "Lakshadweep",
    "NCT of Delhi", "Puducherry",
]

# Utility/DISCOM-specific rows are tagged with a "__UTILITY_" prefix so they
# can be kept separate from state-level totals (avoids double-counting).
# Non-state aggregate rows (Total, India, Others) are tagged "__DROP".
NAME_MAP = {}


def _register(raw_list, canonical):
    for r in raw_list:
        NAME_MAP[r.strip().lower()] = canonical


_register(["Andhra Pradesh"], "Andhra Pradesh")
_register(["Arunachal Pradesh"], "Arunachal Pradesh")
_register(["Assam"], "Assam")
_register(["Bihar", "Bihar (Urban Areas)", "Bihar (Rural Areas)",
            "Bihar (Urban)", "Bihar (Rural)"], "Bihar")
_register(["Chhattisgarh", "Chhattisgrah"], "Chhattisgarh")
_register(["Goa"], "Goa")
_register(["Gujarat", "Gujarat (Urban Areas)", "Gujarat (Rural Areas)",
            "Gujarat (state DISCOMs)"], "Gujarat")
_register(["Gujarat-(Torrent Power Ltd., Ahmedabad)",
            "Gujarat-Torrent Power Ltd. Ahmedabad"], "__UTILITY_Gujarat_Torrent_Ahmedabad")
_register(["Gujarat-(Torrent Power Ltd., Surat)",
            "Gujarat-Torrent Power Ltd. Surat"], "__UTILITY_Gujarat_Torrent_Surat")
_register(["Haryana"], "Haryana")
_register(["Himachal Pradesh"], "Himachal Pradesh")
_register(["Jammu and Kashmir", "Jammu & Kashmir", "J&K", "Jammu Kashmir"], "Jammu and Kashmir")
_register(["Jharkhand", "Jharkhand (Urban Areas)", "Jharkhand (Rural Areas)"], "Jharkhand")
_register(["D.V.C. (Jharkhand Area)", "DVC Jharkhand", "D.V.C.- (Jharkhand)"], "__UTILITY_DVC_Jharkhand")
_register(["Karnataka"], "Karnataka")
_register(["Kerala"], "Kerala")
_register(["Madhya Pradesh", "Madhya Pradesh (Urban Areas)", "Madhya Pradesh (Rural Areas)",
            "MP (Urban)", "MP (Rural)"], "Madhya Pradesh")
_register(["Maharashtra", "Maharashtra-MSEDCL"], "Maharashtra")
_register(["Maharashtra - Mumbai-(B.E.S.T)", "Maharashtra-Mumbai-(BEST)",
            "Mumbai-B.E.S.T", "Mumbai-B.E.S.T (मुंबई बेस्ट)"], "__UTILITY_Mumbai_BEST")
_register(["Maharashtra - Mumbai-(Adani Electricity)", "Mumbai-Adani, TATA Power Co.",
            "Mumbai (Adani Electricity)"], "__UTILITY_Mumbai_Adani")
_register(["Maharashtra - Mumbai-(TATA's)", "Mumbai (TATA)", "Mumbai (TATA's)"], "__UTILITY_Mumbai_TATA")
_register(["Mumbai-Adani / TATA Power"], "__UTILITY_Mumbai_Adani_TATA_combined")
_register(["Manipur"], "Manipur")
_register(["Meghalaya", "Meghalay"], "Meghalaya")
_register(["Mizoram"], "Mizoram")
_register(["Nagaland", "Nagalnd"], "Nagaland")
_register(["Odisha", "Orissa"], "Odisha")
_register(["Punjab"], "Punjab")
_register(["Rajasthan"], "Rajasthan")
_register(["Sikkim"], "Sikkim")
_register(["Tamil Nadu", "Tamilnadu", "Tamil Nad"], "Tamil Nadu")
_register(["Telangana", "Telangan", "Telengana"], "Telangana")
_register(["Tripura"], "Tripura")
_register(["Uttar Pradesh", "Uttar Pradesh (Urban)", "Uttar Pradesh (Rural)",
            "UP (Urban)", "UP (Rural)"], "Uttar Pradesh")
_register(["Uttarakhand", "Uttrakhand"], "Uttarakhand")
_register(["West Bengal", "West Bengal (Urban)", "West Bengal (Rural)",
            "West Bengal-WBSEDCL"], "West Bengal")
_register(["Andaman and Nicobar Islands", "Andaman & Nicobar Islands", "Andaman & Nicobar",
            "Andaman and Nicobar", "A & N Islands", "Andaman Nicobar Island"],
           "Andaman and Nicobar Islands")
_register(["Chandigarh"], "Chandigarh")
_register(["Dadra and Nagar Haveli and Daman and Diu", "Dadra & Nagar Haveli and Daman & Diu",
            "Dadra & Nagar Haveli / Daman & Diu", "Dadra and Nagar Haveli and Daman and Di"],
           "Dadra and Nagar Haveli and Daman and Diu")
_register(["Ladakh"], "Ladakh")
_register(["Lakshadweep"], "Lakshadweep")
_register(["NCT of Delhi", "Delhi", "Delhi (BYPL/BRPL/NDPL-TPDDL & NDMC)",
            "Delhi-(BYPL/BRPL/TPDDL)", "Delhi(NDMC)", "Delhi (NDMC)"], "NCT of Delhi")
_register(["Puducherry", "Pondicherry", "Puducherri"], "Puducherry")
_register(["Total", "India", "INDIA", "Grand Total", "Others", "Others (unspecified)"], "__DROP")


def normalize_state(raw):
    """Map a raw state-name string to its canonical form.

    Returns a string starting with "__UNMAPPED__" if no match is found
    (so unmapped names are visible for debugging, not silently dropped).
    """
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return NAME_MAP.get(key, f"__UNMAPPED__{raw}")
