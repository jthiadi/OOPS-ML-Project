# api.py
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pathlib import Path
from typing import List, Dict, Any, Optional
import sqlite3
import subprocess
import os
import base64
import datetime
import requests
import google.generativeai as genai
from google.generativeai import types

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from google import genai
from google.genai import types

# Replace this with your real API key from Google Generative AI Studio
API_KEY = "FILL_ME"

# Initialize the client with your API key
client = genai.Client(api_key=API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================
# PATHS / CONSTANTS
# ===========================================================
BASE_DIR = Path(__file__).resolve().parent
PYTHON_EXE = BASE_DIR / "venv" / "Scripts" / "python.exe"
DET_LOG = BASE_DIR / "detector_stdout.txt"

DB_PATH = BASE_DIR / "left_items.db"
FRAME_OUTPUT_PATH = BASE_DIR / "live_frame.jpg"
LIVE_LOG_PATH = BASE_DIR / "live_log.txt"

GEMINI_MODEL = "gemini-2.0-flash"
# NOTE: kalau mau pakai env var, ganti jadi:
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "FILL_ME" #Same API as filled above

print("[API] BASE_DIR          =", BASE_DIR)
print("[API] DB_PATH           =", DB_PATH)
print("[API] FRAME_OUTPUT_PATH =", FRAME_OUTPUT_PATH)
print("[API] LIVE_LOG_PATH     =", LIVE_LOG_PATH)

# track main.py process
detector_process: Optional[subprocess.Popen] = None


# ===========================================================
# DB helpers
# ===========================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_exists_or_500():
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail=f"DB not found: {DB_PATH}")


# ===========================================================
# IMAGE PATH RESOLVER (used by /item-image and /describe)
# ===========================================================
def _resolve_image_path(image_path: str) -> str:
    """
    Convert whatever is stored in left_items.image_path into an
    absolute path on disk.

    Handles:
    - absolute paths already (e.g. C:/.../captures/xxx.jpg)
    - relative paths like "captures/xxx.jpg" (older rows)
    """
    if os.path.isabs(image_path):
        return image_path
    return str(DB_PATH.parent / image_path)


# ===========================================================
# Gemini helper
# ===========================================================

def call_gemini_with_image(prompt: str, img_b64: str) -> str:
    """
    Call Gemini with text + image via the SDK and return the generated text.
    """
    logger.info('tes')
    # Convert base64 to bytes
    with open(img_b64, 'rb') as f:
        img_b64 = f.read()

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=img_b64,
                mime_type='image/jpeg'
            ),
            prompt
        ]
    )

    logger.info(f"Generated caption: {response.text}")
    return response.text


# ===========================================================
# FastAPI app
# ===========================================================
app = FastAPI(title="Library Left-Item Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================
# Basic
# ===========================================================
@app.get("/health")
def health():
    return {"status": "ok"}


# ===========================================================
# STATS (Dashboard graphs)
# ===========================================================
@app.get("/stats/30days")
def stats_30days():
    db_exists_or_500()
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DATE(captured_at) AS day, COUNT(*) AS count
            FROM left_items
            WHERE datetime(captured_at) >= datetime('now', '-30 days')
            GROUP BY DATE(captured_at)
            ORDER BY DATE(captured_at)
            """
        )
        rows = cur.fetchall()
        return [{"date": r["day"], "count": r["count"]} for r in rows]
    finally:
        conn.close()


@app.get("/stats/category")
def stats_category():
    db_exists_or_500()
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT item_name, COUNT(*) AS count
            FROM left_items
            WHERE datetime(captured_at) >= datetime('now', '-30 days')
            GROUP BY item_name
            ORDER BY count DESC
            """
        )
        rows = cur.fetchall()
        return [{"item_name": r["item_name"], "count": r["count"]} for r in rows]
    finally:
        conn.close()


# ===========================================================
# ITEMS list / search
# ===========================================================
@app.get("/items/recent")
def recent_items(days: int = 30, limit: int = 50):
    db_exists_or_500()
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, obj_id, item_name, owner_side, table_id,
                   captured_at, image_path, grok_desc
            FROM left_items
            WHERE datetime(captured_at) >= datetime('now', ?)
            ORDER BY datetime(captured_at) DESC
            LIMIT ?
            """,
            (f"-{days} days", limit),
        )
        rows = cur.fetchall()
        result: List[Dict[str, Any]] = []
        for r in rows:
            result.append(
                {
                    "id": r["id"],
                    "obj_id": r["obj_id"],
                    "item_name": r["item_name"] or "",
                    "owner_side": r["owner_side"] or "",
                    "table_id": r["table_id"] or "",
                    "captured_at": r["captured_at"] or "",
                    "image_path": r["image_path"] or "",
                    "grok_desc": r["grok_desc"] or "",
                }
            )

        return result
    finally:
        conn.close()


@app.get("/items/search")
def items_search(
    date: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 500,
):
    """
    Search left_items by optional date + keyword.
    """
    db_exists_or_500()
    conn = get_conn()
    try:
        cur = conn.cursor()
        sql = """
            SELECT id, obj_id, item_name, owner_side, table_id,
                   captured_at, image_path, grok_desc
            FROM left_items
            WHERE 1=1
        """
        params: list[Any] = []

        if date:
            sql += " AND date(captured_at) = date(?) "
            params.append(date)

        if keyword:
            sql += " AND item_name LIKE '%' || ? || '%' "
            params.append(keyword)

        sql += " ORDER BY datetime(captured_at) DESC LIMIT ?"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()
        result: List[Dict[str, Any]] = []
        for r in rows:
            result.append(
                {
                    "id": r["id"],
                    "obj_id": r["obj_id"],
                    "item_name": r["item_name"] or "",
                    "owner_side": r["owner_side"] or "",
                    "table_id": r["table_id"] or "",
                    "captured_at": r["captured_at"] or "",
                    "image_path": r["image_path"] or "",
                    "grok_desc": r["grok_desc"] or "",
                }
            )

        return result
    finally:
        conn.close()


# ===========================================================
# LIVE FRAME & LOGS
# ===========================================================
@app.get("/live-frame")
def live_frame():
    """
    Return the latest camera frame safely.
    Prevents empty/invalid bytes after pressing Q.
    """
    if not FRAME_OUTPUT_PATH.exists():
        raise HTTPException(status_code=404, detail="Live frame not found yet.")

    # ---- NEW SAFETY: check for empty / corrupt file ----
    try:
        size = os.path.getsize(FRAME_OUTPUT_PATH)
        if size < 100:  # too small to be a real JPEG
            raise HTTPException(
                status_code=503,
                detail="Frame temporarily unavailable (camera stopping).",
            )
    except Exception:
        raise HTTPException(status_code=503, detail="Frame unavailable.")

    # ---- Read file fully into memory ----
    try:
        with open(FRAME_OUTPUT_PATH, "rb") as f:
            img_bytes = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed reading frame: {e}")

    return Response(
        content=img_bytes,
        media_type="image/jpeg",
        headers={"Content-Length": str(len(img_bytes))},
    )


@app.get("/logs/live")
def logs_live(lines: int = 200):
    if not LIVE_LOG_PATH.exists():
        return {"lines": []}

    with LIVE_LOG_PATH.open("r", encoding="utf-8") as f:
        all_lines = f.readlines()

    selected = [line.rstrip("\n") for line in all_lines[-lines:]]
    return {"lines": selected}


# ===========================================================
# ITEM IMAGE BY ID (used by Flutter dashboard)
# ===========================================================
@app.get("/item-image/{item_id}")
def get_item_image(item_id: int):
    """
    Return the cropped item image for a given left_items.id.

    Works with:
    - absolute paths saved by new save_left_item_event
    - older relative paths "captures\\xxx.jpg"
    """
    db_exists_or_500()
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT image_path FROM left_items WHERE id = ?",
            (item_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No left_items row found with id={item_id}",
        )

    raw_path = row["image_path"]
    if not raw_path:
        raise HTTPException(
            status_code=404,
            detail=f"Row id={item_id} has empty image_path",
        )

    img_path = _resolve_image_path(raw_path)

    if not os.path.exists(img_path):
        raise HTTPException(
            status_code=404,
            detail=f"Image file not found on server: {img_path}",
        )

    return FileResponse(
        img_path,
        media_type="image/jpeg",
        filename=os.path.basename(img_path),
    )


# ===========================================================
# START / STOP main.py detector
# ===========================================================
@app.get("/detect/status")
def detect_status():
    global detector_process
    running = detector_process is not None and detector_process.poll() is None
    return {"running": running}


@app.post("/detect/start")
def detect_start():
    global detector_process
    # if already running, don't start again
    if detector_process is not None and detector_process.poll() is None:
        return {"running": True, "message": "Already running."}

    script_path = BASE_DIR / "main.py"
    if not script_path.exists():
        raise HTTPException(status_code=500, detail=f"main.py not found at {script_path}")

    print("[API] Starting main.py ...")

    log_file = open(DET_LOG, "w", encoding="utf-8")

    detector_process = subprocess.Popen(
        [PYTHON_EXE, str(script_path)],
        cwd=str(BASE_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return {"running": True, "message": "Detector started."}


@app.post("/detect/stop")
def detect_stop():
    global detector_process
    if detector_process is None or detector_process.poll() is not None:
        detector_process = None
        return {"running": False, "message": "Detector not running."}

    print("[API] Stopping main.py ...")
    detector_process.terminate()
    try:
        detector_process.wait(timeout=5)
    except Exception:
        detector_process.kill()
    detector_process = None
    return {"running": False, "message": "Detector stopped."}


# ===========================================================
# AI DESCRIBE (Gemini) - SAFE VERSION WITH FALLBACK
# ===========================================================

@app.post("/items/{item_id}/describe")
def describe_item(item_id: int):
    """
    Generate (or reuse) an AI description for a left_items row.
    """

    logger.info(f"[describe_item] START item_id={item_id}")

    db_exists_or_500()
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    row = cur.execute(
        """
        SELECT id, item_name, owner_side, table_id,
               captured_at, image_path, grok_desc
        FROM left_items
        WHERE id = ?
        """,
        (item_id,),
    ).fetchone()

    if row is None:
        conn.close()
        logger.warning(f"[describe_item] Item {item_id} not found")
        raise HTTPException(status_code=404, detail=f"Item id={item_id} not found")

    item_name = row["item_name"] or "object"
    owner_side = row["owner_side"] or "unknown"
    table_id = row["table_id"] or "unknown"
    captured = row["captured_at"] or "unknown time"
    existing = row["grok_desc"]

    logger.info(f"[describe_item] Loaded row: item_name={item_name}, table={table_id}, captured={captured}")

    # Fallback description first
    description = (
        f"A {item_name} was left on table {table_id} "
        f"on the {owner_side} side, captured at {captured}."
    )
    logger.info(f"[describe_item] Fallback description prepared")

    # --- Load image if exists ---
    img_b64: Optional[str] = None
    raw_path = row["image_path"]

    if raw_path:
        img_path = _resolve_image_path(raw_path)
        logger.info(f"[describe_item] Resolved image path: {img_path}")

        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                logger.info(f"[describe_item] Loaded image ({len(img_bytes)} bytes)")
            except Exception as e:
                logger.exception(
                    f"[describe_item] Failed reading image for item {item_id}: {e}"
                )
        else:
            logger.warning(f"[describe_item] Image NOT FOUND at path: {img_path}")
    else:
        logger.warning(f"[describe_item] No image_path in database")

    # --- Gemini AI call ---
    if GEMINI_API_KEY and img_b64:
        try:
            logger.info(f"[describe_item] Sending to Gemini...")

            prompt = (
                f'The database log says this forgotten item is: "{item_name}" '
                f'on table {table_id} (owner side: {owner_side}, captured at: {captured}). '
                "Look at the image and ONLY describe extra visual details about this item: "
                "color, shape, material, approximate size, cables, logo, stickers, cracks, dirt, or other unique marks. "
                f"Do NOT change the item type (it is already a {item_name}). "
                "Answer in at most 35 English words."
            )

            ai_text = call_gemini_with_image(prompt, img_path)

            logger.info(f"[describe_item] Gemini response: {ai_text}")

            if ai_text:
                description = ai_text

        except Exception as e:
            logger.exception(f"[describe_item] Gemini describe failed: {e}")
    else:
        if not GEMINI_API_KEY:
            logger.warning("[describe_item] GEMINI_API_KEY missing → fallback used")
        if not img_b64:
            logger.warning(f"[describe_item] No valid image for {item_id} → fallback used")

    # --- Save to DB ---
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[describe_item] Saving description to DB at {ts}")

    cur.execute(
        "UPDATE left_items SET grok_desc = ?, grok_desc_at = ? WHERE id = ?",
        (description, ts, item_id),
    )
    conn.commit()
    conn.close()

    logger.info(f"[describe_item] DONE item_id={item_id}")

    return {"id": item_id, "description": description, "updated_at": ts}



# ===========================================================
# EXPORT PDF
# ===========================================================
@app.get("/items/export/pdf")
def export_pdf(date: Optional[str] = None, keyword: Optional[str] = None):
    """
    Generate simple PDF summary of left_items (filtered by date/keyword).
    Flutter will just open the file in browser.
    """
    db_exists_or_500()
    conn = get_conn()
    try:
        cur = conn.cursor()
        sql = """
            SELECT id, item_name, owner_side, table_id,
                   captured_at, grok_desc
            FROM left_items
            WHERE 1=1
        """
        params: list[Any] = []
        if date:
            sql += " AND date(captured_at) = date(?) "
            params.append(date)
        if keyword:
            sql += " AND item_name LIKE '%' || ? || '%' "
            params.append(keyword)

        sql += " ORDER BY datetime(captured_at) DESC"
        cur.execute(sql, params)
        rows = cur.fetchall()
    finally:
        conn.close()

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = reports_dir / f"left_items_{ts}.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    y = height - 40

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, y, "Left Items Report")
    y -= 24

    c.setFont("Helvetica", 10)
    info = f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    c.drawString(40, y, info)
    y -= 16

    filters = []
    if date:
        filters.append(f"Date = {date}")
    if keyword:
        filters.append(f"Keyword = {keyword}")
    c.drawString(40, y, "Filters: " + (", ".join(filters) if filters else "None"))
    y -= 20

    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, y, "ID")
    c.drawString(70, y, "Item")
    c.drawString(190, y, "Side")
    c.drawString(230, y, "Table")
    c.drawString(270, y, "Captured at")
    y -= 14
    c.setFont("Helvetica", 9)

    for r in rows:
        if y < 40:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 9)

        line_id = str(r["id"])
        c.drawString(40, y, line_id)
        c.drawString(70, y, (r["item_name"] or "")[:20])
        c.drawString(190, y, r["owner_side"] or "")
        c.drawString(230, y, r["table_id"] or "")
        c.drawString(270, y, (r["captured_at"] or "")[:19])
        y -= 12

        desc = (r["grok_desc"] or "").strip()
        if desc:
            if y < 40:
                c.showPage()
                y = height - 40
                c.setFont("Helvetica", 9)
            c.drawString(70, y, "Desc: " + desc[:90])
            y -= 12

    c.save()

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
    )
