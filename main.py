import csv
import os
import re
import time
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from openpyxl import load_workbook
from pydantic import BaseModel

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Thiếu GEMINI_API_KEY trong file .env")

client = genai.Client(api_key=api_key)

app = FastAPI(title="Vinamilk AI Business Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXCEL_PATH = "VINAMILK.xlsx"
PRODUCTS: List[dict] = []


def load_products_from_excel(path: str) -> List[dict]:
    if not os.path.exists(path):
        print(f"Cảnh báo: Không tìm thấy file {path}")
        return []

    wb = load_workbook(filename=path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    headers_raw = next(rows_iter, None)
    if not headers_raw:
        wb.close()
        return []

    headers = [str(h or "").strip() for h in headers_raw]
    items: List[dict] = []

    for r in rows_iter:
        if not r or all(v is None or str(v).strip() == "" for v in r):
            continue
        row_map = {
            headers[i]: str(r[i] if i < len(r) and r[i] is not None else "").strip()
            for i in range(len(headers))
            if headers[i]
        }
        items.append({
            "ten": row_map.get("Tên sản phẩm", ""),
            "nhom": row_map.get("Nhóm sản phẩm", ""),
            "doi_tuong": row_map.get("Đối tượng sử dụng", ""),
            "thanh_phan": row_map.get("Thành phần", ""),
            "dinh_duong": row_map.get("Thông tin dinh dưỡng", ""),
            "huong_dan": row_map.get("Hướng dẫn sử dụng", ""),
            "gia": row_map.get("Giá bán", ""),
            "link": row_map.get("Nguồn thông tin", ""),
        })

    wb.close()
    return items


PRODUCTS = load_products_from_excel(EXCEL_PATH)
print(f"Đã nạp thành công {len(PRODUCTS)} sản phẩm từ {EXCEL_PATH}")


def search_relevant_products(query: str, top_k: int = 8) -> str:
    if not PRODUCTS:
        return "Chưa có dữ liệu sản phẩm từ file Excel."

    tokens = [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 2]
    scored = []
    for p in PRODUCTS:
        searchable = f"{p['ten']} {p['nhom']} {p['doi_tuong']} {p['thanh_phan']} {p['dinh_duong']}".lower()
        name_text = f"{p['ten']} {p['nhom']}".lower()
        score = 0
        for t in tokens:
            if t in name_text:
                score += 4
            elif t in searchable:
                score += 1
        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [item[1] for item in scored[:top_k]] if scored else PRODUCTS[:5]

    blocks = []
    for idx, p in enumerate(selected, 1):
        blocks.append(
            f"--- Sản phẩm {idx}: {p['ten']} ---\n"
            f"- Nhóm: {p['nhom']}\n"
            f"- Đối tượng sử dụng: {p['doi_tuong']}\n"
            f"- Giá bán tham khảo: {p['gia']}\n"
            f"- Thành phần: {p['thanh_phan']}\n"
            f"- Thông tin dinh dưỡng: {p['dinh_duong']}\n"
            f"- Hướng dẫn sử dụng: {p['huong_dan']}\n"
            f"- Link sản phẩm: {p['link']}"
        )
    return "\n\n".join(blocks)


def log_conversation(user_msg: str, bot_reply: str):
    log_file = "chat_history.csv"
    file_exists = os.path.exists(log_file)
    with open(log_file, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Thoi_gian", "Cau_hoi_khach_hang", "Phan_hoi_AI"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_msg, bot_reply])


BASE_SYSTEM_INSTRUCTION = """
Bạn là Trợ lý AI Chăm sóc Khách hàng & Tư vấn Kinh doanh của Vinamilk.
Quy tắc tư vấn:
1. Ưu tiên sử dụng DỮ LIỆU SẢN PHẨM VINAMILK được cung cấp kèm theo câu hỏi để trả lời chính xác tên sản phẩm, giá bán, đối tượng sử dụng, thành phần, giá trị dinh dưỡng, cách pha/sử dụng và gửi kèm link sản phẩm nếu có.
2. Trình bày rõ ràng, thân thiện, có gạch đầu dòng dễ đọc.
3. Nếu sản phẩm khách hỏi không nằm trong danh sách dữ liệu được truy xuất, hãy nói rõ danh mục hiện tại chưa có thông tin chi tiết và gợi ý liên hệ tổng đài Vinamilk 1900 636 979 hoặc website vinamilk.com.vn. Không tự bịa giá bán.
"""

FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]


class ChatMessage(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def serve_homepage():
    return FileResponse("index.html")


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Tin nhắn không được để trống.")

    product_context = search_relevant_products(user_msg, top_k=8)
    full_instruction = (
        f"{BASE_SYSTEM_INSTRUCTION}\n\n"
        f"=== DỮ LIỆU SẢN PHẨM VINAMILK TRUY XUẤT TỪ HỆ THỐNG ===\n"
        f"{product_context}"
    )

    contents = []
    if req.history:
        for item in req.history[-8:]:
            role = "model" if item.role == "model" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=item.text)],
                )
            )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_msg)],
        )
    )

    last_error = None
    for model_name in FALLBACK_MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=full_instruction,
                        temperature=0.3,
                    ),
                )
                reply_text = response.text or "Xin lỗi, hiện tại mình chưa thể phản hồi câu hỏi này."
                log_conversation(user_msg, reply_text)
                return ChatResponse(reply=reply_text)
            except Exception as e:
                last_error = str(e)
                if "503" in last_error or "UNAVAILABLE" in last_error:
                    time.sleep(1.5)
                    continue
                break

    raise HTTPException(
        status_code=503,
        detail=f"Hệ thống AI đang quá tải tạm thời, bạn thử gửi lại sau vài giây nhé. (Chi tiết: {last_error})",
    )