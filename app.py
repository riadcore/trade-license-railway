# Trade License Renew (DNCC/DSCC)
# ✅ Robust OCR + Regex
# ✅ Auto-detect Renew Fee + Signboard Charge (per year)
# ✅ Already Renewed logic
# ✅ Restart button
# ✅ No file downloads
import io, os, re, shutil, pathlib
from datetime import date, datetime
from typing import Dict, Any, Optional, Tuple
import gradio as gr
import fitz  # PyMuPDF
from PIL import Image, ImageFilter, ImageChops, ImageStat, ImageOps, ImageDraw, ImageEnhance
import unicodedata

# --- OCR config (same everywhere: local, Docker, Railway) ---
# You can override this later with an environment variable OCR_CONFIG
OCR_CONFIG = os.getenv("OCR_CONFIG", "--oem 1 --psm 6 -c preserve_interword_spaces=1")
OCR_LANG   = os.getenv("OCR_LANG", "ben+eng")


# --- Environment normalization for Docker / Railway ---
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"
os.environ["LANG"] = "bn_BD.UTF-8"
os.environ["LC_ALL"] = "bn_BD.UTF-8"
os.environ["OMP_THREAD_LIMIT"] = "2"



def bn_norm(s: str) -> str:
    if not s:
        return s
    s = unicodedata.normalize("NFC", s)
    # unify separator variants to colon
    s = s.replace("：", ":").replace("ঃ", ":").replace("।", ":")
    # strip zero-width ghosts and collapse spaces
    s = re.sub(r"[\u200b\u200c\u200d\u2060]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()
# -------- keep storage tiny on Spaces / containers --------
for p in ["~/.cache/huggingface", "~/.cache/pip", "~/.cache/torch", "~/.cache"]:
    pp = pathlib.Path(os.path.expanduser(p))
    if pp.exists():
        shutil.rmtree(pp, ignore_errors=True)
# -------- OCR (for scanned PDFs / images) --------
try:
    import pytesseract
    # Point to tesseract.exe on Windows
    if os.name == "nt":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
FAST_OCR_CONFIG = r"--oem 1 --psm 6 -l ben+eng"
FALLBACK_OCR_CONFIGS = [
    r"--oem 1 --psm 11 -l ben+eng",  # sparse text
    r"--oem 1 --psm 4 -l ben+eng",   # block/column layout
    r"--oem 1 --psm 7 -l ben+eng",   # single text line (good for small rows/labels)
]
def _save_debug_image(im: Image.Image, prefix="debug_preprocessed"):
    """Save OCR-ready grayscale/thresholded image with timestamp for visual debugging."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ✅ Change this path to your desired folder
    debug_dir = os.environ.get("DEBUG_DIR", "./debug_images")
    # Create folder if not exists
    os.makedirs(debug_dir, exist_ok=True)
    out_path = os.path.join(debug_dir, f"{prefix}_{ts}.jpg")
    try:
        im.save(out_path, "JPEG")
        print(f"[DEBUG] Saved OCR view → {out_path}")
    except Exception as e:
        print(f"[WARN] Could not save debug image: {e}")
def _prep_image_for_ocr(im: Image.Image) -> Image.Image:
    orig_w, orig_h = im.size  # 👈 store original size before resize
    # 1) Blue suppression before grayscale
    im = im.convert("RGB")
    r, g, b = im.split()
    if ImageStat.Stat(b).mean[0] - max(ImageStat.Stat(r).mean[0], ImageStat.Stat(g).mean[0]) > 12:
        im = Image.blend(r, g, 0.5)
        im = ImageChops.subtract(im, b.point(lambda x: x // 10))
    im = im.convert("L")
  
    # 2) Adaptive upscale for small scans
    scale = 1.6 if orig_w <= 800 else 1.4
    im = im.resize((int(im.width * scale), int(im.height * scale)))

    # 3) Gentler threshold to keep thin Bangla text alive
    im = im.point(lambda x: 255 if x > 210 else (0 if x < 120 else x))

    # 4) Clamp max width
    max_w = 2200
    if im.width > max_w:
        h = int(im.height * (max_w / im.width))
        im = im.resize((max_w, h))
    # 5) Mask QR/photo areas (from 08_app.py)
    h, w = im.height, im.width
    if h > w and h >= 1200 and w < 1600:
        draw = ImageDraw.Draw(im)
        draw.rectangle([int(0.04*w), int(0.06*h), int(0.22*w), int(0.20*h)], fill=255)
        draw.rectangle([int(0.74*w), int(0.05*h), int(0.95*w), int(0.23*h)], fill=255)
    # 6) Save debug (optional)

    # Paper Tree: gentle re-binarize + single thicken (fast + stable)
    if orig_w <= 800 and orig_h <= 1100:
        # softer binarization
        im = im.point(lambda x: 255 if x > 215 else (0 if x < 115 else x))
        # slightly thicker (3×3 kernel)
        im = im.filter(ImageFilter.MinFilter(3))
        # auto-invert if background is darker
        if ImageStat.Stat(im).mean[0] < 128:
            im = ImageOps.invert(im)
        # 📸 Paper Tree: tiny autocontrast (keeps thin Bangla strokes)
        if orig_w <= 800 and orig_h <= 1100:
            im = ImageOps.autocontrast(im, cutoff=2)

    # ✅ Only save debug images when debugging is turned on
    #if os.getenv("TL_DEBUG", "0") == "1":
    #_save_debug_image(im, "preprocessed")

    # --- camera-shadow rescue: run only if top↔bottom brightness differs a lot
    t_mean = ImageStat.Stat(im.crop((0, 0, im.width, int(im.height*0.25)))).mean[0]
    b_mean = ImageStat.Stat(im.crop((0, int(im.height*0.75), im.width, im.height))).mean[0]
    if abs(t_mean - b_mean) > 18:
        # flatten illumination via blurred background removal
        bg = im.filter(ImageFilter.GaussianBlur(9))
        im = ImageChops.subtract(im, ImageChops.subtract(bg, Image.new("L", im.size, int(min(t_mean, b_mean)))))
        im = ImageOps.autocontrast(im, cutoff=1)

    return im
def ocr_image_fast(im: Image.Image, config: str = FAST_OCR_CONFIG) -> str:
    if not OCR_AVAILABLE:
        return ""
    im = _prep_image_for_ocr(im)
    txt = pytesseract.image_to_string(im, lang=OCR_LANG, config=OCR_CONFIG) or ""
    # ✅ If the first OCR result already contains license hints, stop here (saves time)
    low = txt.lower()
    if ("trad" in low and ("dncc" in low or "dscc" in low)) or ("লাইসেন্স নং" in low):
        return txt

    small = (im.width * im.height) <= (800 * 1100 * 1.5)
    # If tiny image already yielded enough text with digits, stop early (fast)
    if small and len(txt.strip()) >= 80 and any(ch.isdigit() for ch in txt):
        return txt

    # 📍 If first OCR attempt is too weak, retry with broader configs
    if len(txt.strip()) < 60 or not any(ch.isdigit() for ch in txt):
        cfgs = (
            FALLBACK_OCR_CONFIGS if not small else [
                r"--oem 1 --psm 7 -l ben+eng",  # text line mode - catches labels
                r"--oem 1 --psm 6 -l ben+eng",  # uniform block - good for tables
                r"--oem 1 --psm 4 -l ben+eng",  # column layout - rescues multi-column rows
            ]
        )
        for cfg in cfgs:
            t2 = pytesseract.image_to_string(im, lang=OCR_LANG, config=cfg) or ""
            if len(t2.strip()) > len(txt.strip()):
                txt = t2

    # 📍 Mini-ROI OCR pass: fee table zone (broader coverage + 3-pass)
    # ✅ Only run 3× ROI OCR if license hints were NOT found yet
    if ("trad" not in txt.lower()) and ("লাইসেন্স নং" not in txt) and small and len(txt.strip()) < 100:

        h, w = im.height, im.width

        # Pass 1: main fee band (mid → low)
        y0, y1 = int(h * 0.60), int(h * 0.90)
        band1 = im.crop((int(w * 0.05), y0, int(w * 0.95), y1))
        t1 = pytesseract.image_to_string(band1, lang=OCR_LANG, config="--oem 1 --psm 6") or ""

        # Pass 2: sli
        # ghtly higher band (some templates sit higher)
        y0b, y1b = int(h * 0.45), int(h * 0.60)
        band2 = im.crop((int(w * 0.05), y0b, int(w * 0.95), y1b))
        t2 = pytesseract.image_to_string(band2, lang=OCR_LANG, config="--oem 1 --psm 6") or ""

        # Pass 3: even lower band (catches signboard totals)
        y0c, y1c = int(h * 0.86), int(h * 0.95)
        band3 = im.crop((int(w * 0.05), y0c, int(w * 0.95), y1c))
        t3 = pytesseract.image_to_string(band3, lang=OCR_LANG, config="--oem 1 --psm 6") or ""

        # Use the longest of the three
        t_roi = max([t1, t2, t3], key=lambda t: len(t.strip()))
        if len(t_roi.strip()) > len(txt.strip()):
            txt = txt + "\n" + t_roi

    # 📍 Header ROI: catch "TRAD/DNCC/..." & top labels if missing
    if ("trad" not in txt.lower()) and ("লাইসেন্স নং" not in txt) and (small and len(txt.strip()) < 180):
        h, w = im.height, im.width
        y0h, y1h = int(h * 0.16), int(h * 0.34)
        head = im.crop((int(w * 0.08), y0h, int(w * 0.92), y1h))
        th = pytesseract.image_to_string(head, lang=OCR_LANG, config="--oem 1 --psm 7") or ""

        if len(th.strip()) > 10 and len(th.strip()) > len(txt.strip()):
            txt = txt + "\n" + th

    if ("trad" not in txt.lower()) and ("লাইসেন্স নং" not in txt) and small and len(txt.strip()) < 120:
        best_txt = txt
        best_score = len(txt.strip()) + sum(ch.isdigit() for ch in txt) * 3
        for ang in (-1.2, 1.2):   # was (-2.0, -1.2, -0.6, 0.6, 1.2, 2.0)
            imp = im.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=255)
            t = pytesseract.image_to_string(imp, lang=OCR_LANG, config=OCR_CONFIG) or ""

            score = len(t.strip()) + sum(ch.isdigit() for ch in t) * 3
            if score > best_score:
                best_txt, best_score = t, score
        txt = best_txt

    # 📍 Tiny rotate retry if text is still too short after fallbacks
    if not small and len(txt.strip()) < 40:
        for ang in (-0.7, 0.7):
            imp = im.rotate(ang, expand=False, fillcolor=255)
            t3 = pytesseract.image_to_string(imp, lang=OCR_LANG, config=OCR_CONFIG) or ""

            if len(t3.strip()) > len(txt.strip()):
                txt = t3

    # --- if text still weak, try a slightly wider deskew sweep
    if len(txt.strip()) < 120:
        best = txt
        for ang in (-2.2, -1.6, -0.9, -0.4, 0.4, 0.9, 1.6, 2.2):
            imp = im.rotate(ang, resample=Image.BICUBIC, expand=False, fillcolor=255)
            t2 = pytesseract.image_to_string(imp, config=config) or ""
            if len(t2.strip()) > len(best.strip()):
                best = t2
        if len(best.strip()) > len(txt.strip()):
            txt = best

    # --- if still weak and looks like a phone shot, try micro-perspective fixes
    if len(txt.strip()) < 120 and im.height > im.width:
        def _try_warp(dx_px: int):
            w, h = im.size
            # move top-left/right inward and bottom-left/right outward (and the inverse)
            dst = (dx_px, 0,  w - dx_px, 0,  w + dx_px, h,  -dx_px, h)
            warped = im.transform((w, h), Image.QUAD, dst, Image.BICUBIC)
            return pytesseract.image_to_string(warped, lang=OCR_LANG, config=OCR_CONFIG) or ""

        for dx in (int(im.width*0.015), int(im.width*0.03)):
            for s in (+dx, -dx):
                t2 = _try_warp(s)
                if len(t2.strip()) > len(txt.strip()):
                    txt = t2
    return txt
def extract_text_from_path(path: str, ocr_all_pages: bool = False) -> Tuple[str, str]:
    """Return (text, method). Prefer selectable text, else OCR."""
    p = pathlib.Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        try:
            with fitz.open(path) as doc:
                # 1) selectable text fast path
                txt = []
                for pg in doc:
                    t = pg.get_text("text")
                    if t:
                        txt.append(t)
                if any(len(t) > 30 for t in txt):
                    all_txt = "\n".join(txt).strip()

                    # ✅ Only return if text looks strong (long enough and has digits)
                    if len(all_txt) >= 400 and re.search(r"[০-৯0-9]{3,}", all_txt):
                        return all_txt, "pymupdf"
                    # ❌ else: DO NOT return — just fall through to the OCR section below

                # 2) OCR fallback (higher DPI for sharper numerals)
                pages = range(len(doc)) if ocr_all_pages else range(min(1, len(doc)))
                ocr_chunks = []
                for i in pages:
                    pix = doc[i].get_pixmap(dpi=220, alpha=False)  # try 300 if needed
                    im = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_chunks.append(ocr_image_fast(im))
                return "\n".join(ocr_chunks).strip(), "ocr_image"
        except Exception:
            pass
    # Image branch
    try:
        im = Image.open(path)
        return ocr_image_fast(im), "ocr_image"
    except Exception:
        return "", "none"
# =========================
# PARSERS
# =========================
# ---------- Generic, brand-agnostic cleanups for Business Name ----------
BN_CONSONANTS = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"
BN_VOWEL_SIGNS = "\u09BE\u09BF\u09C0\u09C1\u09C2\u09C7\u09C8\u09CB\u09CC"  # া ি ী ু ূ ে ৈ ো ৌ

def _normalize_biz_name_generic(s: str) -> str:

    if not s:
        return s
    # base normalize + your bn_norm (already imported above)
    s = bn_norm(s)
    # remove zero-width ghosts (if any survived)
    s = re.sub(r"[\u200b\u200c\u200d\u2060]", "", s)
    # drop leading numbering/bullets/separators (common in left table)
    s = re.sub(r"^\s*[০-৯0-9]+[).:-]*\s*", "", s)
    s = re.sub(r"^[\s:：ঃ।=\/\-–—]+", "", s)
    # collapse extra spaces
    s = re.sub(r"\s{2,}", " ", s).strip()
    # fix halant + spaces between consonants → join (ক্ ম → ক্ম)
    s = re.sub(
    rf"([{BN_CONSONANTS}])\u09CD\s+([{BN_CONSONANTS}])",
    lambda m: m.group(1) + "\u09CD" + m.group(2),
    s
)
    # conservative de-halant only when a vowel sign follows (keeps letters)
    # s = re.sub(
    #     rf"([{BN_CONSONANTS}])\u09CD([মনরলয])([{BN_VOWEL_SIGNS}])",
    #     r"\1\2\3",
    #     s
    # )
    # keep legal suffixes, just normalize spacing before them
    # s = re.sub(r"\s*(লিঃ|লিমিটেড|Limited|Ltd\.?)\s*$", r" \1", s, flags=re.IGNORECASE)
    
    # add a space before legal suffixes if it’s missing (does NOT change the suffix text)
    s = re.sub(r"(?<!\s)(লিঃ|লি[:：]?|লিমিটেড|Limited|Ltd\.?)\s*$", r" \1", s, flags=re.IGNORECASE)
    # if two words glued and the second begins with an independent vowel (e.g., আ/ই/উ/এ/ও), insert one space
    s = re.sub(rf"([{BN_CONSONANTS}{BN_VOWEL_SIGNS}])([অআইঈউঊএঐওঔআ])", r"\1 \2", s)
    # if there is still no space at all, add one soft split at a common boundary like 'র' + consonant
    if " " not in s:
        s = re.sub(r"(র)([কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ])", r"\1 \2", s, count=1)

    return s.strip()
# ---- Business name (Bangla exact labels) ----
BUSINESS_NAME_LABELS = [
    r"১[।.]?\s*ব্যবসা\s*প্রতিষ্ঠানের\s*নাম",
    r"ব্যবসা\s*প্রতিষ্ঠানের\s*নাম",
    r"প্রতিষ্ঠানের\s*নাম",
]
def extract_business_name(text: str) -> str:
    """
    Finds '১। ব্যবসা প্রতিষ্ঠানের নাম' (or variants) and returns the name.
    Works when the name is on the same line after ':' or on the next line(s).
    """
    if not text:
        return ""
    name = "" 
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        for lbl in BUSINESS_NAME_LABELS:
            if re.search(lbl, ln):
                # Try SAME LINE first
                tail = re.split(lbl, ln, flags=re.IGNORECASE, maxsplit=1)
                cand = ""
                if len(tail) > 1:
                    cand = tail[1].strip()
                def _clean(s: str) -> str:
                    s = bn_norm(s)
                    # strip leading separators and bullets
                    s = re.sub(r"^[\s:।–—\-]+", "", s)
                    s = re.sub(r"^[০-৯0-9]+[).:-]*\s*", "", s)
                    # brand-agnostic Bengali cleanup (does NOT remove 'লিঃ' / 'লিমিটেড' / 'Ltd.')
                    s = _normalize_biz_name_generic(s)
                    return s.strip()
                if cand and not re.fullmatch(r"[:।–—\-]*", cand):
                    name = _clean(cand)
                    if name:
                        return name
                # Otherwise look in next few lines
                for j in range(i + 1, min(i + 6, len(lines))):
                    s = lines[j].strip()
                    if not s or re.fullmatch(r"[:।–—\-]*", s):
                        continue
                    name = _clean(s)
                    if name:
                        return name
    # If still not found, try fallback search across entire text
    if not name:
        m = re.search(r"ব্যবসা\s*প্রতিষ্ঠানের\s*নাম[:：ঃ।=–—-]*\s*([\u0980-\u09FF A-Za-z]+)", text)
        if m:
            return _normalize_biz_name_generic(m.group(1).strip())
    return ""
# --- Corp detection helpers ---
_ZW = re.compile(r"[\u200b\u200c\u200d\u2060]")   # zero-widths
def _clean_bn(s: str) -> str:
    s = normalize_digits(s or "")
    s = _ZW.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()
# Variants: কর্পোরেশন / করপোরেশন; allow extra spaces/newlines between words
DNCC_PATTERNS = [
    r"ঢাকা\s*উত্তর\s*সিটি\s*ক(?:র্?প)?পোরেশন",
    r"\bdncc\b", r"dncc\.gov\.bd", r"www\.dncc\.gov\.bd",
    r"\bDhaka\s*North\s*City\s*Corporation\b",
]
DSCC_PATTERNS = [
    r"ঢাকা\s*দক্ষিণ\s*সিটি\s*ক(?:র্?প)?পোরেশন",
    r"\bdscc\b", r"dscc\.gov\.bd", r"www\.dscc\.gov\.bd",
    r"\bDhaka\s*South\s*City\s*Corporation\b",
]
def detect_corporation(text: str) -> str:
    t = _clean_bn(text).lower()
    score = {"DNCC": 0, "DSCC": 0}
    # Strong signal: license number prefix
    m = re.search(r"\btrad\/(dscc|dncc)\/\d{3,}\/\d{4}\b", t, flags=re.I)
    if m:
        score[m.group(1).upper()] += 5
    # Headers / body text / URLs
    for pat in DNCC_PATTERNS:
        if re.search(pat, t, flags=re.I):
            score["DNCC"] += 2
    for pat in DSCC_PATTERNS:
        if re.search(pat, t, flags=re.I):
            score["DSCC"] += 2
    # Extra nudge if the lone words appear near 'সিটি'
    if re.search(r"উত্তর\s*সিটি", t): score["DNCC"] += 1
    if re.search(r"দক্ষিণ\s*সিটি", t): score["DSCC"] += 1
    if score["DNCC"] > score["DSCC"]:
        return "DNCC"
    if score["DSCC"] > score["DNCC"]:
        return "DSCC"
    return "UNKNOWN"
def extract_license_number(text: str) -> Optional[str]:
    """
    Robustly parse TRAD/<DSCC|DNCC>/<number>/<year> even if OCR:
    - drops slashes or turns them into I|l|1
    - inserts spaces
    - reads 'DSCCI' or 'DNCCI' etc.
    """
    if not text:
        return None
    t = bn_norm(text)
    t = normalize_digits(t)
    # Accept /, |, \, I, l, 1 (common OCR confusions) as separators, optional
    SEP = r"[\/|\\Il1\-\s]*"
    # TRAD + corp (DSCC/DNCC with optional trailing I) + id + year
    # Example matches:
    #   TRADDSCC0079072025
    #   TRAD/DSCC/007907/2025
    #   TRAD I DSCCI I 007907 I 2025
    pat = rf"T\s*R\s*A\s*D{SEP}(D\s*[SN]\s*C\s*C[Ii]?){SEP}(\d{{3,}}){SEP}(20\d{{2}})"
    m = re.search(pat, t, flags=re.IGNORECASE)
    if not m:
        # 0) find the same-line "tail" right after the license label (Bangla or English)
        tail = ""
        m2 = re.search(r"লাইসেন্স\s*নং[:：ঃ।=–—-]*([^\n]{0,120})", t, flags=re.IGNORECASE)
        if m2:
            tail = m2.group(1)
        else:
            m3 = re.search(r"License\s*No[:：:=\- ]*([^\n]{0,120})", t, flags=re.IGNORECASE)
            if m3:
                tail = m3.group(1)

        if tail:
            tail = normalize_digits(tail)

            # 1) Prefer a proper TRAD/DSCC|DNCC/<5–7 digits>/<year> if present in the tail
            m_trad = re.search(
                r"TRAD[^\n]{0,20}(DSCC|DNCC)[^\n]{0,10}(\d{5,7})[^\n]{0,8}(20\d{2})",
                tail, flags=re.IGNORECASE
            )
            if m_trad:
                corp_fix = m_trad.group(1).upper()
                num_fix  = m_trad.group(2)   # keeps leading zeroes (e.g., 040052)
                year_fix = m_trad.group(3)
                return f"TRAD/{corp_fix}/{num_fix}/{year_fix}"

            # 2) Otherwise: choose the 5–7 digit block nearest *before* the year on that line
            y_m = re.search(r"(20\d{2})", tail)
            nums = list(re.finditer(r"(\d{5,7})", tail))
            if y_m and nums:
                y_pos = y_m.start()
                left = [n for n in nums if n.start() < y_pos]
                pick = (sorted(left, key=lambda n: y_pos - n.start())[0]
                        if left else nums[-1])
                num  = pick.group(1)
                year = y_m.group(1)
                corp_guess = ("DNCC" if re.search(r"DNCC|উত্তর", t, flags=re.IGNORECASE)
                            else ("DSCC" if re.search(r"DSCC|দক্ষিণ", t, flags=re.IGNORECASE) else "DNCC"))
                return f"TRAD/{corp_guess}/{num}/{year}"

        return None

    corp_raw = re.sub(r"\s+", "", m.group(1)).upper()      # e.g. "DSCCI"
    corp = "DSCC" if corp_raw.startswith("DSCC") else "DNCC"
    num  = m.group(2)
    year = m.group(3)
    return f"TRAD/{corp}/{num}/{year}"
def extract_last_renew_year(text: str) -> Optional[str]:
    m = re.search(r"(20\d{2})\s*[-–]\s*(20\d{2})", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b == a + 1:
            return f"{a}-{b}"
    return None
def extract_valid_until(text: str) -> Optional[str]:
    pats = [
        r"মেয়াদ.*?(?:৩০|30)\s*(?:জুন|June)[,]?\s*(\d{4})",
        r"(?:valid\s*until|validity).*?30\s*(?:June)[,]?\s*(\d{4})",
        r"(?:জুন|June)[,]?\s*(\d{4}).*?(?:পর্যন্ত|until)"
    ]
    for pat in pats:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            year = normalize_digits(m.group(1))   # 👈 force English digits
            return f"30 June {year}"
    years = re.findall(r"(20\d{2})", normalize_digits(text))  # 👈 normalize here too
    if years:
        return f"30 June {max(map(int, years))}"
    return None
def infer_last_renew_from_valid_until(valid_until: Optional[str]) -> Optional[str]:
    if not valid_until:
        return None
    m = re.search(r"(\d{4})$", valid_until.strip())
    if not m:
        return None
    end = int(m.group(1))
    return f"{end-1}-{end}"
def extract_per_year_fees(text: str) -> Tuple[Optional[float], Optional[float]]:
    # Renew fee: allow optional 'লাইসেন্স', slash/dash, both য়/য় forms, and ন/ণ
    renew_labels = [
        r"(?:লাইসেন্স\s*[/\-]?\s*)?নব[াা]\s*[য়য়য]?\s*[নণ]\s*ফি",  # লাইসেন্স নবায়ন/নবায়ন ফি
        r"নবায়ন\s*ফি",
        r"নবায়ন\s*ফি",
        r"(?:renew\s*fee|govt\s*renew\s*fee)",  # English fallback, if ever
    ]
    sign_labels = [
        r"সাইনবোর্ড\s*কর\s*\(পরিচিতিমূলক\)",
        r"সাইনবোর্ড\s*কর",
        r"সাইনবোর্ড\s*চার্জ",
        r"সাইন\s*বোর্ড\s*কর",
        r"সাইনবোর্ড\s*কর\s*\(বার্ষিক\)",
        r"Signboard\s*(?:Charge|Tax)(?:\s*\(per\s*year\))?",
        r"Sign\s*Board\s*(?:Charge|Tax)",
        r"বোর্ড\s*চার্জ", 
    ]
 
    renew_val = extract_amount_by_labels(text, renew_labels)
    sign_val = extract_amount_by_labels(text, sign_labels)

    # 🩹 Avoid false duplication: if both are equal and renew_val is found,
    # force signboard to None so UI shows blank instead of a duplicate.
    if renew_val is not None and sign_val == renew_val:
        sign_val = None

    return renew_val, sign_val

# Bangla → Latin digits
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
def normalize_digits(s: Optional[str]) -> Optional[str]:
    return s.translate(BN_DIGITS) if s else s
def _to_float_safe(num_str: Optional[str]) -> Optional[float]:
    """
    Converts a captured money-like string to float.
    Handles Bangla digits, commas, hidden chars, zero-width separators, and digit runs.
    """
    if not num_str:
        return None
    orig = normalize_digits(num_str)
    # 1) Remove common noise
    cleaned = re.sub(r"[^\d.]", "", orig)
    if cleaned:
        try:
            return float(cleaned)
        except Exception:
            pass
    # 2) Join ALL digits, even across broken runs (handles 5​000 → 5000)
    digits_only = "".join(re.findall(r"\d", orig))
    if digits_only and len(digits_only) >= 3:
        try:
            return float(digits_only)
        except Exception:
            return None

    return None
def extract_amount_by_labels(text: str, label_patterns) -> Optional[float]:
    """
    Capture number near label (same/next few lines). Handles Bangla digits, currency signs,
    and separators like :, -, =, '/=' between label and number. Picks the FIRST plausible match.
    More tolerant to OCR that splits label/number across lines.
    """
    t = normalize_digits(bn_norm(text or ""))
    money_core = r"(?:\d[\d,，\s\u200b\u200c\u200d\u2060]*\d(?:\.\d+)?|\d+)"
    money = rf"(?:৳|Tk\.?|Taka|টাকা)?\s*{money_core}"
    sep = r"(?:\s*[:=–—\-\/=]\s*)?"
    # ✅ Add this line:
    is_signboard = any(("সাইনবোর্ড" in p) or ("Signboard" in p) for p in label_patterns)
    
    # ✅ Add this tiny helper function (below is_signboard)
    def _looks_like_year(s: str) -> bool:
        d = "".join(re.findall(r"\d", s or ""))  # extract only digits from the string
        return len(d) == 4 and 1900 <= int(d) <= 2100
    
    # ---- Signboard-specific fast detection ----
    if is_signboard:
        sign_lbl = re.compile(
            r"(সাইন\s*বোর্ড|সাইনবোর্ড|বোর্ড\s*চার্জ|বোর্ড\s*কর)\s*(কর|চার্জ)?|Sign\s*board|Signboard",
            flags=re.IGNORECASE
        )
        stop_words = re.compile(
            r"(ভ্যাট|VAT|Form|ফর্ম|Service|সার্ভিস|Source|সূত্র|Book|বই|Bank|ব্যাংক|Total|সর্বমোট)",
            flags=re.IGNORECASE
        )

        # ✅ local plausibility so zero/tiny numbers are ignored
        def _plausible_fast(v: Optional[float]) -> bool:
            return v is not None and 200 <= v <= 50000

        def _pick_rightmost(chunk: str) -> Optional[float]:
            m = re.findall(money_core, chunk)
            for cand in reversed(m):
                # avoid picking a year by mistake (helper defined above)
                if not _looks_like_year(cand):
                    val = _to_float_safe(cand)
                    if _plausible_fast(val):     # ✅ enforce plausibility here
                        return val
            return None

        lines = normalize_digits(bn_norm(text or "")).splitlines()

        for i, ln in enumerate(lines):
            if not sign_lbl.search(ln):
                continue

            # (A) Try SAME line (rightmost, only if plausible)
            val = _pick_rightmost(ln)
            if val is not None:
                return val

            # (B) Try next 1–3 CONTENT lines, skipping blanks & stop-words  ✅
            taken = []
            k = i + 1
            while k < len(lines) and len(taken) < 3:
                s = lines[k].strip()
                if s and not re.fullmatch(r"[\s:：ঃ।=\/\-–—]*", s) and not stop_words.search(s):
                    taken.append(s)
                k += 1

            if taken:
                blob = " ".join(taken)
                val2 = _pick_rightmost(blob)
                if val2 is not None:
                    return val2
            
    def _plausible(v: Optional[float]) -> bool:
        return v is not None and 200 <= v <= 50000
    lines = t.splitlines()
    for i, ln in enumerate(lines):
        for lbl in label_patterns:
            if not re.search(lbl, ln, flags=re.IGNORECASE):
                continue
            # ---- SAME LINE
            m = re.search(lbl + sep + r"[^\d০-৯]{0,32}?" + money, ln, flags=re.IGNORECASE)
            if m:
                for cand in reversed(re.findall(money_core, m.group(0))):
                    val = _to_float_safe(cand)
                    if _plausible(val):
                        return val

            # ---- SKIP punctuation-only lines after the label
            j = i + 1
            while j < len(lines) and re.fullmatch(r"[\s:：ঃ।=\/\-–—]*", lines[j]):
                j += 1
            # ---- NEXT CONTENT LINE
            if j < len(lines):
                m2 = re.search(money, lines[j], flags=re.IGNORECASE)
                if m2:
                    for cand in reversed(re.findall(money_core, m2.group(0))):
                        val2 = _to_float_safe(cand)
                        if _plausible(val2):
                            return val2

            # ---- GUARDED 2-LINE LOOKAHEAD (Signboard only, avoid VAT/others)
            if is_signboard:
                stop_words = r"(ভ্যাট|VAT|Form|ফর্ম|Service|সার্ভিস|Source|সূত্র|Book|বই|Bank|ব্যাংক|Total|সর্বমোট)"
                block = []
                k = j
                while k < len(lines) and len(block) < 2:
                    s = lines[k].strip()
                    if s and not re.fullmatch(r"[\s:：ঃ।=\/\-–—]*", s):
                        block.append(s)
                    k += 1
                if block:
                    blob = " ".join(block)
                    mX = re.search(money, blob, flags=re.IGNORECASE)
                    if mX:
                        for cand in reversed(re.findall(money_core, mX.group(0))):
                            valX = _to_float_safe(cand)
                            if _plausible(valX):
                                return valX

            # ✅ Only do this multi-line search if NOT signboard
            if not is_signboard:
                k = j
                taken = []
                while k < len(lines) and len(taken) < 3:
                    if not re.fullmatch(r"[\s:：ঃ।=\/\-–—]*", lines[k]):
                        taken.append(lines[k].strip())
                    k += 1
                if taken:
                    blob = " ".join(taken)
                    m3 = re.search(money, blob, flags=re.IGNORECASE)
                    if m3:
                        for cand in reversed(re.findall(money_core, m3.group(0))):
                            val3 = _to_float_safe(cand)
                            if _plausible(val3):
                                return val3
    return None
# =========================
# DUE & FINE
# =========================
def current_fy_end_year(today: date) -> int:
    return today.year + 1 if today.month >= 7 else today.year
def parse_last_renew_end_year(lbl: str) -> Optional[int]:
    m = re.match(r"\s*(20\d{2})\s*[-–]\s*(20\d{2})\s*$", lbl)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return b if b == a + 1 else None
def compute_due(last_renew_lbl: str, today: Optional[date] = None) -> int:
    today = today or date.today()
    cur_end = current_fy_end_year(today)
    last_end = parse_last_renew_end_year(last_renew_lbl)
    if last_end is None:
        return 1
    return max(1, cur_end - last_end)
def months_since_july(today: Optional[date] = None) -> int:
    today = today or date.today()
    m = today.month - 7
    return m if m > 0 else 0
def compute_fine_months(due_years: int, today: Optional[date] = None) -> int:
    today = today or date.today()
    msj = months_since_july(today)
    add = msj if msj > 2 else 0
    return max(0, (due_years * 12) - 12 + add)
# =========================
# CALCULATORS
# =========================
def calc_dncc(last_renew_lbl, renew_fee_py, signboard_py, source_tax_py=3000.0,
              service=500.0, others=500.0, book=270.0, bank=50.0) -> Dict[str, float]:
    due = compute_due(last_renew_lbl)
    fine_m = compute_fine_months(due)
    govt_renew = renew_fee_py * due
    signboard  = signboard_py * due
    source_tax = source_tax_py * due
    total_fine = (renew_fee_py * 0.10) * fine_m
    vat = 0.15 * (govt_renew + signboard + total_fine)
    total_govt = govt_renew + signboard + source_tax + total_fine + vat
    grand_total = total_govt + service + others + book + bank
    return {
        "Due (years)": float(due),
        "Fine for Month (auto)": float(fine_m),
        "Govt Renew Fee": float(govt_renew),
        "Signboard Charge": float(signboard),
        "Source TAX": float(source_tax),
        "Total Fine": float(total_fine),
        "VAT (15%)": float(vat),
        "Total Govt. Fees": float(total_govt),
        "Service Charge": float(service),
        "Others": float(others),
        "Book Charge": float(book),
        "Bank Charge": float(bank),
        "Grand Total": float(grand_total),
    }
def calc_dscc(last_renew_lbl, renew_fee_py, signboard_py, source_tax_py=3000.0,
              form_fee=50.0, service=1000.0, bank=50.0) -> Dict[str, float]:
    due = compute_due(last_renew_lbl)
    fine_m = compute_fine_months(due)
    govt_renew = renew_fee_py * due
    signboard  = signboard_py * due
    source_tax = source_tax_py * due
    total_fine = (renew_fee_py * 0.10) * fine_m
    vat = 0.15 * (govt_renew + signboard + total_fine)
    total_govt = govt_renew + signboard + source_tax + total_fine + vat + form_fee
    grand_total = total_govt + service + bank
    return {
        "Due (years)": float(due),
        "Fine for Month (auto)": float(fine_m),
        "Govt Renew Fee": float(govt_renew),
        "Signboard Charge": float(signboard),
        "Source TAX": float(source_tax),
        "Total Fine": float(total_fine),
        "VAT (15%)": float(vat),
        "Form Fee": float(form_fee),
        "Total Govt. Fees": float(total_govt),
        "Service Charge": float(service),
        "Bank Charge": float(bank),
        "Grand Total": float(grand_total),
    }
# =========================
# UI HELPERS
# =========================
def fmt_taka(x: float) -> str:
    try:
        return f"৳{int(x):,}" if float(x).is_integer() else f"৳{x:,.2f}"
    except Exception:
        return str(x)
BN_LABELS = {
    "Due (years)": "বকেয়া বছর",
    "Fine for Month (auto)": "মোট জরিমানার মাসের সংখ্যা",
    "Govt Renew Fee": "সরকারি নবায়ন ফি",
    "Signboard Charge": "সাইনবোর্ড চার্জ",
    "Source TAX": "সূত্র কর",
    "Total Fine": "মোট জরিমানা",
    "VAT (15%)": "ভ্যাট (১৫%)",
    "Total Govt. Fees": "মোট সরকারি ফি",
    "Service Charge": "সেবামূল্য",
    "Others": "অন্যান্য",
    "Book Charge": "বইয়ের মূল্য",
    "Bank Charge": "ব্যাংক চার্জ",
    "Form Fee": "ফর্ম ফি",
    "Grand Total": "সর্বমোট"
}
def breakdown_to_html_bn(corp: str, lic: str, biz: str,
                         last_renew: str, due: int, bd: dict,
                         renew_py: Optional[float], sign_py: Optional[float]) -> str:
    """
    Builds Bangla HTML breakdown table.
    Changes:
      - Header label: টাকা → ফলাফল
      - Rows 'বকেয়া বছর' & 'মোট জরিমানার মাসের সংখ্যা' show plain numbers (no ৳)
    """
    # Keys that should display without currency symbol
    PLAIN_KEYS = {"Due (years)", "Fine for Month (auto)"}

    def _fmt_cell(k: str, v: float) -> str:
        # omit "৳" only for the two keys above
        if k in PLAIN_KEYS:
            try:
                return str(int(float(v)))  # just numeric
            except Exception:
                return str(v)
        else:
            return fmt_taka(v)  # keep ৳ for all others
    rows = []
    for k, v in bd.items():
        if k == "Grand Total":
            continue
        label = BN_LABELS.get(k, k)
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td style='text-align:right'>{_fmt_cell(k, v)}</td></tr>"
        )
    grand = (
        f"<tr style='font-weight:700;border-top:2px solid #444'>"
        f"<td>{BN_LABELS.get('Grand Total','Grand Total')}</td>"
        f"<td style='text-align:right'>{fmt_taka(bd.get('Grand Total',0))}</td></tr>"
    )
    extra = ""
    if renew_py is not None or sign_py is not None:
        extra = (
            "<div style='margin:.35rem 0 .2rem;color:#666;font-size:13px'>"
            f"Per-year (detected): Renew = {fmt_taka(renew_py or 0)}, "
            f"Signboard = {fmt_taka(sign_py or 0)}</div>"
        )
    hdr = f"""
    <div style="font-family:Inter,system-ui,Segoe UI,Arial; line-height:1.35; max-width:760px;">
      <h3 style="margin:.2rem 0;">ট্রেড লাইসেন্স নবায়ন — {corp}</h3>
      <div style="font-size:14px;color:#888;">
        লাইসেন্স: {lic or '-'} &nbsp;•&nbsp;
        প্রতিষ্ঠানের নাম: {biz or '-'} &nbsp;•&nbsp;
        শেষ নবায়ন বছর: {last_renew or '-'} &nbsp;•&nbsp;
        বকেয়া বছর: {due}
      </div>
      {extra}
      <table style="width:100%; border-collapse:collapse; margin-top:.5rem;">
        <thead>
        <tr>
            <th style="padding:8px; background:#111; color:#fff !important; font-weight:600; text-align:left;">
            আইটেম
            </th>
            <th style="padding:8px; background:#111; color:#fff !important; font-weight:600; text-align:right;">
            ফলাফল
            </th>
        </tr>
        </thead>

        <tbody>
          {''.join(rows)}
          {grand}
        </tbody>
      </table>
    </div>
    """
    return hdr
def parse_valid_until_date(valid_until_label: Optional[str]) -> Optional[date]:
    if not valid_until_label:
        return None
    s = valid_until_label.strip().replace(",", "")
    try:
        return datetime.strptime(s, "%d %B %Y").date()
    except Exception:
        m = re.search(r"(20\d{2})", s)
        if m:
            return date(int(m.group(1)), 6, 30)
    return None
# =========================
# PIPELINE
# =========================
def analyze_upload(file_path: str) -> Dict[str, Any]:
    # 1) Try selectable text first (fast path if doc has good text)
    text, method = extract_text_from_path(file_path, ocr_all_pages=False)
    print("\n\n===== EXTRACTED TEXT (DEBUG) =====\n", text[:3000], "\n\n")
    corp = detect_corporation(text)
    lic  = extract_license_number(text)
    valid_until = extract_valid_until(text)
    last_renew = extract_last_renew_year(text) or infer_last_renew_from_valid_until(valid_until)
    biz_name = extract_business_name(text)
    # 2) Auto-detect per-year fees from the current text
    renew_py, sign_py = extract_per_year_fees(text)

    # 🩹 PDF-only rescue (even if we later switched to image OCR): catch Signboard on the next lines
    is_pdf = str(file_path).lower().endswith(".pdf")
    if is_pdf and sign_py is None:
        t = normalize_digits(bn_norm(text or ""))
        lines = t.splitlines()

        money_core = r"(?:\d[\d,，\s\u200b\u200c\u200d\u2060]*\d(?:\.\d+)?|\d+)"
        stop_words = re.compile(r"(ভ্যাট|VAT|Form|ফর্ম|Service|সার্ভিস|Source|সূত্র|Book|বই|Bank|ব্যাংক|Total|সর্বমোট)", flags=re.IGNORECASE)

        def _looks_like_year(s: str) -> bool:
            d = "".join(re.findall(r"\d", s or ""))
            return len(d) == 4 and 1900 <= int(d) <= 2100

        def _plausible(v):
            return v is not None and 200 <= v <= 50000

        for i, ln in enumerate(lines):
            ln2 = re.sub(r"সাইন\s*বোর্ড", "সাইনবোর্ড", ln, flags=re.IGNORECASE)
            if re.search(r"সাইনবোর্ড\s*(?:কর|চার্জ)\s*(?:\(পরিচিতিমূলক\))?", ln2, flags=re.IGNORECASE):
                # try same line, rightmost number
                m_same = re.findall(money_core, ln2)
                for cand in reversed(m_same or []):
                    if not _looks_like_year(cand):
                        val = _to_float_safe(cand)
                        if _plausible(val):
                            sign_py = val
                            break
                if sign_py is not None:
                    break
                # look 1–3 content lines below, skipping blanks & stop-words
                taken = []
                k = i + 1
                while k < len(lines) and len(taken) < 3:
                    s = lines[k].strip()
                    if s and not re.fullmatch(r"[\s:：ঃ।=\/\-–—]*", s) and not stop_words.search(s):
                        taken.append(s)
                    k += 1
                blob = " ".join(taken)
                m_next = re.findall(money_core, blob)
                for cand in reversed(m_next or []):
                    if not _looks_like_year(cand):
                        val = _to_float_safe(cand)
                        if _plausible(val):
                            sign_py = val
                            break
                if sign_py is not None:
                    break

    # 🧷 Ultra-narrow PDF fallback: same-line "… : 480" capture, if still None
    if is_pdf and sign_py is None:
        tt = normalize_digits(bn_norm(text or ""))
        m = re.search(r"(সাইন\s*বোর্ড|সাইনবোর্ড)[^\n]{0,40}[:：=]\s*(\d{2,6})", tt, flags=re.IGNORECASE)
        if m:
            v = _to_float_safe(m.group(2))
            if v is not None and 200 <= v <= 50000:
                sign_py = v

    # 3) 🔄 Prefer OCR if the PyMuPDF text looks incomplete
    if method == "pymupdf":
        weak = (
            (renew_py is None and sign_py is None) or
            (not lic) or
            (not biz_name) or
            (not valid_until) or
            (corp == "UNKNOWN")
        )
        if weak:
            # 📍 Instead of OCRing the whole PDF pages, rasterize the first page and run through ocr_image_fast()
            import fitz
            with fitz.open(file_path) as doc:
                pg = doc[0]
                pix = pg.get_pixmap(dpi=300)  # rasterize at good OCR DPI
                from PIL import Image
                import io
                im = Image.open(io.BytesIO(pix.tobytes("png")))

            text = ocr_image_fast(im)  # ✅ run your best OCR pipeline here
            method = "ocr_image"       # mark that we used image OCR

            # re-parse fields again based on the new text
            corp = detect_corporation(text)
            lic = extract_license_number(text)
            valid_until = extract_valid_until(text)
            last_renew = extract_last_renew_year(text) or infer_last_renew_from_valid_until(valid_until)
            biz_name = extract_business_name(text)
            r2, s2 = extract_per_year_fees(text)
            if r2 is not None: renew_py = r2
            if s2 is not None: sign_py = s2

    # 3) ⛑️ OCR fallback:
    # If BOTH amounts are missing and the method was selectable text ("pymupdf"),
    # rasterize first page again and use image OCR instead of ocr_pdf
    if renew_py is None and sign_py is None and method == "pymupdf":
        import fitz
        with fitz.open(file_path) as doc:
            pg = doc[0]
            pix = pg.get_pixmap(dpi=300)
            from PIL import Image
            import io
            im = Image.open(io.BytesIO(pix.tobytes("png")))
        text_ocr = ocr_image_fast(im)
        r2, s2 = extract_per_year_fees(text_ocr)
        if r2 is not None or s2 is not None:
            text = text_ocr
            renew_py = r2 if r2 is not None else renew_py
            sign_py  = s2 if s2 is not None else sign_py
            method = "ocr_image"  # ✅ force image OCR label

    # 4) Remove default fallback — only keep OCR-detected values
    # If renew_py or sign_py is not found, they will stay as None (or blank)
    # This ensures the app only depends on real OCR extraction
    # and does not auto-fill anything.
    # 5) Due/fine summary (same as before)
    due = compute_due(last_renew) if last_renew else 1
    fine_m = compute_fine_months(due)

    # 🛡️ Final PDF guard: if Signboard is None or 0.0, try strict same-line capture again
    try:
        is_pdf = str(file_path).lower().endswith(".pdf")
    except Exception:
        is_pdf = False

    if is_pdf and (sign_py is None or sign_py == 0.0):
        tt = normalize_digits(bn_norm(text or ""))
        # strict: label on the same line followed by : or = and the amount (2–6 digits)
        m = re.search(r"(সাইন\s*বোর্ড|সাইনবোর্ড)[^\n]{0,80}[:：=]\s*(\d{2,6})", tt, flags=re.IGNORECASE)
        if m:
            v = _to_float_safe(m.group(2))
            if v is not None and 200 <= v <= 50000:
                sign_py = v

    print("[DEBUG] sign_py →", sign_py, "method:", method)

    return {
        "corporation": corp,
        "license_no": lic or "",
        "valid_until": valid_until or "",
        "last_renew_year": last_renew or "",
        "due_years": due,
        "fine_months": fine_m,
        "renew_fee_py": float(renew_py) if renew_py is not None else None,
        "signboard_py": float(sign_py) if sign_py is not None else None,
        "raw_text": text[:3000],
        "method": method,
        "business_name": biz_name or "",
    }
# =========================
# UI
# =========================
def build_ui():
    with gr.Blocks(title="Trade License Renewal Fee AI Software") as demo:
        gr.Markdown("## Trade License Renewal Fee AI Software")
        with gr.Row():
            file_in = gr.File(label="Upload Trade License File", type="filepath")
           
        analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Row():
            corp_out = gr.Textbox(label="Detected City Corporation", interactive=True)
            license_out = gr.Textbox(label="License No.", interactive=True)
            biz_in = gr.Textbox(label="Business Name", interactive=True)
        with gr.Row():
            valid_out = gr.Textbox(label="Valid Until", interactive=True)
            last_renew_out = gr.Textbox(label="Last Renew Year (YYYY-YYYY)", interactive=True)
            due_out = gr.Number(label="Due (years)", interactive=False, precision=0)
        fine_months_out = gr.Number(label="Fine for Month (auto)", interactive=False, precision=0)
        method_out = gr.Textbox(label="Extraction Method", interactive=False)
        gr.Markdown("### Detected (auto)")
        with gr.Row():
            renew_py_view = gr.Number(label="Renew Fee (per year)", interactive=True, precision=0)
            sign_py_view = gr.Number(label="Signboard Charge (per year)", interactive=True, precision=0)
        with gr.Accordion("Show extracted text (debug)", open=False):
            raw_text_view = gr.Textbox(lines=8, interactive=False)
        gr.Markdown("### Enter/Adjust Other Fees")
        dncc_group = gr.Group(visible=False)
        with dncc_group:
            gr.Markdown("#### Dhaka North City Corporation (DNCC)")
            with gr.Row():
                dncc_src   = gr.Number(label="Source TAX (per year)", value=3000.0, precision=2)
                dncc_service = gr.Number(label="Service Charge", value=500.0, precision=2)
                dncc_others  = gr.Number(label="Others", value=500.0, precision=2)
            with gr.Row():
                dncc_book    = gr.Number(label="Book Charge", value=270.0, precision=2)
                dncc_bank    = gr.Number(label="Bank Charge", value=50.0, precision=2)
        dscc_group = gr.Group(visible=False)
        with dscc_group:
            gr.Markdown("#### Dhaka South City Corporation (DSCC)")
            with gr.Row():
                dscc_src   = gr.Number(label="Source TAX (per year)", value=3000.0, precision=2)
                dscc_form  = gr.Number(label="Form Fee", value=50.0, precision=2)
                dscc_service= gr.Number(label="Service Charge", value=1000.0, precision=2)
            with gr.Row():
                dscc_bank   = gr.Number(label="Bank Charge", value=50.0, precision=2)
        with gr.Row():
            compute_btn = gr.Button("Compute Grand Total", variant="primary")
            restart_btn = gr.Button("Restart", variant="secondary")
        total_out = gr.Number(label="Grand Total (৳)", precision=2, interactive=False)
        with gr.Accordion("Show Detailed Breakdown (raw)", open=False):  # 👈 collapsible
            breakdown_json = gr.JSON(label="Detailed Breakdown (raw)")
        breakdown_html = gr.HTML(label="সারাংশ / Status")
        # Hidden states
        state_corp = gr.State("")
        state_renew_lbl = gr.State("")
        state_license = gr.State("")
        state_valid = gr.State("")
        state_renew_py = gr.State(0.0)
        state_sign_py = gr.State(0.0)
        # ---- Analyze ----
        def on_analyze(file_path: str):
            info = analyze_upload(file_path)
            corp = info["corporation"]
            dncc_vis = (corp == "DNCC")
            dscc_vis = (corp == "DSCC")
            return (
                info["corporation"], info["license_no"],
                info.get("business_name", ""),   # 👈 now auto-fills Business Name
                info["valid_until"], info["last_renew_year"], info["due_years"],
                info["fine_months"], info["method"],
                float(info["renew_fee_py"] or 0.0), float(info["signboard_py"] or 0.0),
                info.get("raw_text", ""),
                dncc_vis and gr.update(visible=True) or gr.update(visible=False),
                dscc_vis and gr.update(visible=True) or gr.update(visible=False),
                # states
                corp, info["last_renew_year"], info["license_no"], info["valid_until"],
                float(info["renew_fee_py"] or 0.0), float(info["signboard_py"] or 0.0),
                # clear outputs
                gr.update(value=""), gr.update(value=0.0),
            )
        analyze_btn.click(
            on_analyze,
            inputs=[file_in],
            outputs=[
                corp_out, license_out, biz_in,
                valid_out, last_renew_out, due_out,
                fine_months_out, method_out,
                renew_py_view, sign_py_view,
                raw_text_view,
                dncc_group, dscc_group,
                state_corp, state_renew_lbl, state_license, state_valid,
                state_renew_py, state_sign_py,
                breakdown_html, total_out
            ],
        )
        # ---- Compute ----
        def on_compute(corp: str, last_lbl: str, lic: str, valid_lbl: str, biz_manual: str,
                       renew_py: float, sign_py: float,
                       dncc_src, dncc_serv, dncc_oth, dncc_book, dncc_bank,
                       dscc_src2, dscc_form, dscc_serv, dscc_bank):
            # If valid until is today or in future -> already renewed
            vdt = parse_valid_until_date(valid_lbl)
            today = date.today()
            if vdt is not None and vdt >= today:
                msg_html = (
                    "<div style='font-family:Inter,system-ui; padding:.6rem; "
                    "border:1px solid #444; background:#111; color:#fff; border-radius:8px;'>"
                    "<b style='color:#00ff88;'>✅ Already Trade License Renewed</b><br>"
                    f"<span style='color:#ccc;'>Valid Until: {valid_lbl or '-'}</span>"
                    "</div>"
                )
                return (
                    0.0,
                    {"status": "Already Trade License Renewed", "valid_until": valid_lbl},
                    gr.update(value=msg_html),
                )
            if not corp or corp == "UNKNOWN":
                msg = {"error": "City corporation not detected. Upload a clearer license."}
                return 0.0, msg, gr.update(value="<i>No data</i>")
            if not last_lbl:
                msg = {"error": "Last renew year missing. Make sure it’s visible on the license."}
                return 0.0, msg, gr.update(value="<i>No data</i>")
            if corp == "DNCC":
                bd = calc_dncc(
                    last_lbl,
                    renew_fee_py=renew_py, signboard_py=sign_py,
                    source_tax_py=dncc_src, service=dncc_serv, others=dncc_oth,
                    book=dncc_book, bank=dncc_bank
                )
            else:
                bd = calc_dscc(
                    last_lbl,
                    renew_fee_py=renew_py, signboard_py=sign_py,
                    source_tax_py=dscc_src2, form_fee=dscc_form, service=dscc_serv, bank=dscc_bank
                )
            total = bd["Grand Total"]
            html = breakdown_to_html_bn(
                corp=corp, lic=lic, biz=biz_manual, last_renew=last_lbl,
                due=int(bd.get("Due (years)", 1)), bd=bd,
                renew_py=renew_py, sign_py=sign_py
            )
            return total, bd, gr.update(value=html)
        compute_btn.click(
            on_compute,
            inputs=[
                corp_out, last_renew_out, license_out, valid_out, biz_in,   # 👈 visible + editable
                renew_py_view, sign_py_view,
                # DNCC others:
                dncc_src, dncc_service, dncc_others, dncc_book, dncc_bank,
                # DSCC others:
                dscc_src, dscc_form, dscc_service, dscc_bank,
            ],
            outputs=[total_out, breakdown_json, breakdown_html],
        )
        # ---- Restart ----
        def on_restart():
            return (
                gr.update(value=None),  # file_in
                gr.update(value=""),    # corp_out
                gr.update(value=""),    # license_out
                gr.update(value=""),    # biz_in
                gr.update(value=""),    # valid_out
                gr.update(value=""),    # last_renew_out
                gr.update(value=0),     # due_out
                gr.update(value=0),     # fine_months_out
                gr.update(value=""),    # method_out
                gr.update(value=None),  # renew_py_view
                gr.update(value=None),  # sign_py_view
                gr.update(value=""),    # raw_text_view
                gr.update(visible=False),  # dncc_group
                gr.update(visible=False),  # dscc_group
                "", "", "", "", 0.0, 0.0,   # states
                gr.update(value=""), gr.update(value=0.0),  # breakdown_html, total_out
            )
        restart_btn.click(
            on_restart,
            inputs=[],
            outputs=[
                file_in,
                corp_out, license_out, biz_in,
                valid_out, last_renew_out, due_out,
                fine_months_out, method_out,
                renew_py_view, sign_py_view,
                raw_text_view,
                dncc_group, dscc_group,
                state_corp, state_renew_lbl, state_license, state_valid, state_renew_py, state_sign_py,
                breakdown_html, total_out
            ]
        )
    return demo
if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False, show_error=True)