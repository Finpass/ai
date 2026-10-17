from src.features.profile.facts import Facts

DISCLAIMER = {
    "id": "Ini panduan edukasi, bukan nasihat investasi.",
    "en": "This is financial guidance and education, not investment advice.",
}

PROFILE = {
    "id": {"Conservative": "Konservatif", "Moderate": "Moderat", "Aggressive": "Agresif"},
    "en": {"Conservative": "Conservative", "Moderate": "Moderate", "Aggressive": "Aggressive"},
}
CATEGORY = {
    "id": {
        "Really Struggling": "Sangat Kesulitan",
        "Struggling": "Kesulitan",
        "Coping": "Cukup Bertahan",
        "Thriving": "Berkembang",
    },
    "en": {c: c for c in ("Really Struggling", "Struggling", "Coping", "Thriving")},
}
DIM = {
    "id": {
        "Security": "Keamanan",
        "Safety": "Keselamatan",
        "Growth": "Pertumbuhan",
        "Wisdom": "Wawasan",
        "Freedom": "Kebebasan",
    },
    "en": {d: d for d in ("Security", "Safety", "Growth", "Wisdom", "Freedom")},
}
LADDER = {
    "id": {
        1: "membangun dana darurat. Artinya, fondasi Anda saat ini adalah punya cadangan kas "
        "yang cukup sebelum mulai berinvestasi.",
        2: "melunasi utang berbunga tinggi. Artinya, dana darurat Anda sudah cukup, jadi "
        "langkah terbaik adalah mengurangi beban utang sebelum berinvestasi lebih jauh.",
        3: "mulai berinvestasi secara dasar. Artinya, kondisi dasar Anda sudah sehat dan Anda "
        "siap membangun portofolio yang terdiversifikasi sedikit demi sedikit.",
        4: "mengoptimalkan investasi. Artinya, Anda sudah punya portofolio yang beragam dan "
        "saatnya meninjau pajak, menambah kontribusi, dan menata ulang alokasi.",
        5: "menjaga dan mewariskan kekayaan. Artinya, aset Anda sudah besar dan fokusnya "
        "beralih ke perencanaan warisan serta efisiensi pajak.",
    },
    "en": {
        1: "building an emergency fund. This means your foundation right now is a solid cash "
        "reserve before you start investing.",
        2: "paying off high-interest debt. This means your emergency fund is in place, so "
        "reducing debt comes before investing more.",
        3: "starting basic investing. This means your basics are healthy and you are ready to "
        "build a diversified portfolio step by step.",
        4: "optimizing your investments. This means you already hold a diversified portfolio "
        "and it is time to review taxes, raise contributions and rebalance.",
        5: "preserving and passing on wealth. This means your assets are substantial and the "
        "focus shifts to estate planning and tax efficiency.",
    },
}


def num(x: float, loc: str) -> str:
    s = f"{x:.0f}" if abs(x - round(x)) < 0.05 else f"{x:.1f}"
    return s.replace(".", ",") if loc == "id" else s


def rp(x: float, loc: str) -> str:
    s = f"{round(x):,}"
    return "Rp " + (s.replace(",", ".") if loc == "id" else s)


def _target(f: Facts) -> str:
    if f.target_min == f.target_max:
        return num(f.target_min, f.locale)
    lo, hi = num(f.target_min, f.locale), num(f.target_max, f.locale)
    return f"{lo} - {hi}"


def _next_step(f: Facts) -> str:
    loc = f.locale
    surplus = f.surplus > 0
    if f.level == 1:
        if loc == "id":
            base = (
                f"Langkah berikutnya: lengkapi dana darurat. Targetnya {num(f.target_min, loc)} "
                f"bulan pengeluaran ({rp(f.target_amount, loc)}), dan saat ini masih kurang "
                f"{rp(f.gap, loc)} (Anda punya {num(f.months, loc)} bulan)."
            )
            tail = (
                f" Sisihkan sebagian surplus bulanan {rp(f.surplus, loc)} ke tabungan terpisah."
                if surplus
                else " Tinjau pengeluaran agar ada ruang untuk menabung setiap bulan."
            )
        else:
            base = (
                f"Next step: complete your emergency fund. The target is "
                f"{num(f.target_min, loc)} months of expenses ({rp(f.target_amount, loc)}), and "
                f"you are still short by {rp(f.gap, loc)} (you hold {num(f.months, loc)} months)."
            )
            tail = (
                f" Set aside part of your monthly surplus of {rp(f.surplus, loc)} in a "
                "separate savings account."
                if surplus
                else " Review your spending so there is room to save every month."
            )
        return base + tail
    if f.level == 2:
        if loc == "id":
            return (
                "Langkah berikutnya: lunasi utang berbunga tinggi lebih dulu. Cicilan Anda "
                f"saat ini {rp(f.installment, loc)} per bulan atau {num(f.dti_pct, loc)}% dari "
                "penghasilan; arahkan surplus bulanan ke utang dengan bunga paling tinggi."
            )
        return (
            "Next step: pay off your highest-interest debt first. Your installments are "
            f"{rp(f.installment, loc)} per month, or {num(f.dti_pct, loc)}% of income; direct "
            "your monthly surplus to the debt with the highest interest."
        )
    if f.level == 3:
        if loc == "id":
            return (
                f"Langkah berikutnya: mulai berinvestasi secara rutin dengan sebagian surplus "
                f"bulanan Anda ({rp(f.surplus, loc)}), sesuaikan dengan profil "
                f"{PROFILE[loc][f.profile]}, dan sebar ke beberapa jenis aset."
            )
        return (
            "Next step: start investing regularly with part of your monthly surplus "
            f"({rp(f.surplus, loc)}), match it to your {PROFILE[loc][f.profile]} profile, and "
            "spread it across several asset types."
        )
    if f.level == 4:
        if loc == "id":
            return (
                f"Langkah berikutnya: tinjau alokasi dan naikkan kontribusi bulanan. Progres "
                f"Anda menuju level berikutnya {num(f.progress, loc)}%."
            )
        return (
            "Next step: review your allocation and raise your monthly contribution. Your "
            f"progress toward the next level is {num(f.progress, loc)}%."
        )
    if loc == "id":
        return (
            "Langkah berikutnya: susun rencana warisan dan efisiensi pajak bersama perencana "
            "keuangan berlisensi."
        )
    return "Next step: draft an estate and tax-efficiency plan with a licensed financial planner."


def render(f: Facts) -> dict[str, str]:
    loc = f.locale
    prof, cat = PROFILE[loc][f.profile], CATEGORY[loc][f.category]
    strong, weak = DIM[loc][f.strongest], DIM[loc][f.weakest]
    n = num
    if loc == "id":
        who = f"{f.name}, " if f.name else ""
        summary = (
            f"{who}berdasarkan jawaban Anda, profil risiko Anda {prof}, posisi Anda di Financial "
            f"Ladder ada di level {f.level}, dan kesehatan finansial Anda berada di kategori "
            f"{cat} dengan skor {n(f.total, loc)}."
        )
        risk = f"Profil risiko Anda adalah {prof}."
        if f.provisional:
            risk += (
                " Profil ini masih bersifat sementara dan dapat berubah seiring data Anda "
                "bertambah."
            )
        if f.capacity is not None and f.willingness is not None:
            risk += (
                f" Skor kemampuan menanggung risiko {n(f.capacity, loc)} dan skor kemauan "
                f"mengambil risiko {n(f.willingness, loc)}."
            )
        ladder = f"Anda berada di Financial Ladder level {f.level}: {LADDER[loc][f.level]}"
        health = (
            f"Skor kesehatan finansial Anda {n(f.total, loc)} dari 100, kategori {cat}. "
            f"Sisi terkuat Anda adalah {strong} ({n(f.dims[f.strongest], loc)}), sedangkan sisi "
            f"yang paling perlu perhatian adalah {weak} ({n(f.dims[f.weakest], loc)})."
        )
    else:
        who = f"{f.name}, " if f.name else ""
        summary = (
            f"{who}based on your answers, your risk profile is {prof}, you are at level "
            f"{f.level} of the Financial Ladder, and your financial health falls in the {cat} "
            f"category with a score of {n(f.total, loc)}."
        )
        risk = f"Your risk profile is {prof}."
        if f.provisional:
            risk += " This profile is provisional and may change as your data grows."
        if f.capacity is not None and f.willingness is not None:
            risk += (
                f" Your risk capacity score is {n(f.capacity, loc)} and your risk willingness "
                f"score is {n(f.willingness, loc)}."
            )
        ladder = f"You are at Financial Ladder level {f.level}: {LADDER[loc][f.level]}"
        health = (
            f"Your financial health score is {n(f.total, loc)} out of 100, category {cat}. "
            f"Your strongest area is {strong} ({n(f.dims[f.strongest], loc)}), while the area "
            f"needing the most attention is {weak} ({n(f.dims[f.weakest], loc)})."
        )
    return {
        "summary": summary,
        "risk": risk,
        "ladder": ladder,
        "health": health,
        "nextStep": _next_step(f),
    }
