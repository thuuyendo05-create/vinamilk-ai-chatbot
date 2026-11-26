import os
import re
import csv
import unicodedata
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Vinamilk AI Customer Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXCEL_FILE = "VINAMILK.xlsx"
LOG_FILE = "chat_history.csv"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

MODEL_CANDIDATES = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
]


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_product_links(excel_path: str) -> List[dict]:
    """Chỉ trích xuất tên sản phẩm và link dẫn tới trang sản phẩm từ file Excel."""
    if not os.path.exists(excel_path):
        return []
    try:
        df = pd.read_excel(excel_path).fillna("")
    except Exception:
        return []

    records = []
    for _, row in df.iterrows():
        name = ""
        url = ""
        for col in df.columns:
            col_norm = normalize_text(col)
            val = str(row[col]).strip()
            if not val:
                continue
            if not url and ("http://" in val or "https://" in val):
                match = re.search(r"https?://[^\s)\]]+", val)
                if match:
                    url = match.group(0)
            if not name and any(k in col_norm for k in ["ten", "san pham", "product", "name"]):
                if not val.startswith("http"):
                    name = val
        if not name:
            for col in df.columns:
                val = str(row[col]).strip()
                if val and not val.startswith("http") and len(val) > 3:
                    name = val
                    break
        if name and url:
            records.append({
                "name": name,
                "norm": normalize_text(name),
                "url": url,
            })
    return records


PRODUCT_LINKS = load_product_links(EXCEL_FILE)


def find_matching_links(query: str, limit: int = 5) -> List[dict]:
    q_norm = normalize_text(query)
    if not q_norm or not PRODUCT_LINKS:
        return []
    tokens = [t for t in q_norm.split() if len(t) >= 2 and t not in {"sua", "vinamilk", "gia", "bao", "nhieu", "loai", "cho", "cua", "tu", "van"}]
    if not tokens:
        tokens = [t for t in q_norm.split() if len(t) >= 2]

    scored = []
    for item in PRODUCT_LINKS:
        score = sum(1 for tk in tokens if tk in item["norm"])
        if q_norm in item["norm"]:
            score += 3
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]


def log_chat(user_msg: str, bot_reply: str, model_used: str):
    file_exists = os.path.exists(LOG_FILE)
    try:
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "model", "user_message", "bot_reply"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), model_used, user_msg, bot_reply])
    except Exception:
        pass


SYSTEM_INSTRUCTION = """
Bạn là Trợ lý AI Tư vấn Khách hàng chính thức của Vinamilk (website: https://www.vinamilk.com.vn/).

Nguyên tắc trả lời:
1. Trả lời tự nhiên, lịch sự, chính xác dựa trên kiến thức về sản phẩm, dinh dưỡng và thông tin công bố trên website chính thức của Vinamilk (https://www.vinamilk.com.vn/).
2. Tuyệt đối KHÔNG tự bịa số liệu chi tiết nếu không chắc chắn; với giá bán hoặc quy cách đóng gói, nêu mức tham khảo theo các dòng phổ biến trên vinamilk.com.vn và lưu ý giá có thể thay đổi tùy kênh bán/chương trình khuyến mãi.
3. Trình bày mạch lạc, dễ đọc:
   - Mở đầu bằng 1 câu chào/dẫn ngắn gọn.
   - Khi giới thiệu sản phẩm, ghi rõ tên sản phẩm trên một dòng riêng, bên dưới chia rõ các mục:
     Đối tượng sử dụng: ...
     Giá bán tham khảo: ... (nếu có nhiều quy cách thì ngăn cách bằng dấu "/" để giao diện tự tách dòng)
     Đặc điểm / Thành phần chính: ...
     Hướng dẫn sử dụng: ...
   - Nếu được cung cấp danh sách link sản phẩm khớp từ hệ thống bên dưới, hãy gắn đúng link đó ở cuối mỗi sản phẩm tương ứng (mỗi link trên 1 dòng riêng). Nếu không có link khớp trong danh sách, gắn link danh mục chính thức https://www.vinamilk.com.vn/
4. Trả lời bằng tiếng Việt rõ ràng, không dùng ký tự trang trí rườm rà.
""".strip()


class HistoryItem(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[HistoryItem]] = []


@app.get("/")
async def serve_home():
    return FileResponse("index.html")


@app.get("/health")
async def health_check():
    return {"status": "ok", "linked_products": len(PRODUCT_LINKS)}


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    user_message = (payload.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Tin nhắn không được để trống.")

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Chưa cấu hình biến môi trường GEMINI_API_KEY trên server.")

    matched_links = find_matching_links(user_message, limit=5)
    link_hint = ""
    if matched_links:
        lines = [f"- {m['name']}: {m['url']}" for m in matched_links]
        link_hint = (
            "\n\n[Danh sách link sản phẩm tham chiếu từ hệ thống — chỉ dùng đúng đường link này nếu nhắc tới sản phẩm tương ứng, KHÔNG lấy thông tin mô tả khác ngoài link]:\n"
            + "\n".join(lines)
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = []
    for item in (payload.history or [])[-6:]:
        role = "user" if item.role == "user" else "model"
        if item.text:
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=item.text)]))

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message + link_hint)],
        )
    )

    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.4,
                    max_output_tokens=1000,
                ),
            )
            reply_text = (response.text or "").strip()
            if reply_text:
                log_chat(user_message, reply_text, model_name)
                return {"reply": reply_text, "model": model_name}
        except Exception as exc:
            last_error = str(exc)
            continue

    fallback_reply = (
        "Chào bạn, hiện hệ thống AI của Vinamilk đang tạm thời bận. "
        "Bạn có thể tham khảo trực tiếp danh mục sản phẩm và thông tin dinh dưỡng mới nhất tại trang chủ chính thức:\n"
        "https://www.vinamilk.com.vn/"
    )
    if matched_links:
        links_str = "\n".join(f"{m['name']}\n{m['url']}" for m in matched_links[:3])
        fallback_reply = (
            "Chào bạn, hiện trợ lý AI đang quá tải nhẹ. Dưới đây là các đường dẫn sản phẩm Vinamilk liên quan để bạn xem nhanh:\n"
            + links_str
        )
    log_chat(user_message, f"[FALLBACK] {last_error}", "fallback")
    return {"reply": fallback_reply, "model": "fallback"}