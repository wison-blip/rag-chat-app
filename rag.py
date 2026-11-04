"""RAG の処理: 資料の読み込み → 分割 → Chroma に登録 → 会話履歴つきで検索して回答。

Lesson 13 の構成（create_history_aware_retriever + create_stuff_documents_chain + create_retrieval_chain）
を、アップロードされた資料向けにまとめたもの。
"""

import uuid

import pymupdf
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHUNKS = 1500  # 埋め込みの料金と時間を抑えるための上限
HISTORY_LIMIT = 10  # チェーンに渡す直近のメッセージ数
BATCH_SIZE = 100

CONTEXTUALIZE_PROMPT = (
    "チャット履歴と、最新のユーザーの質問があります。最新の質問が履歴の内容を前提にしている場合は、"
    "履歴がなくても意味が通じる、独立した質問に書き直してください。"
    "質問には答えず、書き直した質問だけを出力してください。前提にしていなければ、そのまま出力してください。"
)

QA_PROMPT = (
    "あなたは、アップロードされた資料をもとに質問に答えるアシスタントです。\n"
    "- 下の「資料の抜粋」に書かれている内容だけを根拠に、日本語で分かりやすく答えてください。\n"
    "- 抜粋に答えが書かれていないときは、推測せず「資料には書かれていませんでした」と答えてください。\n"
    "- 数字、日付、固有名詞は、抜粋の表記のまま正確に書いてください。\n\n"
    "資料の抜粋:\n{context}"
)


class RagError(Exception):
    """画面にそのまま表示できる文で、資料の処理に失敗したことを伝える。"""


# ---------- 1. 資料の読み込み ----------
def _decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp932"):  # 日本語のテキストは UTF-8 か Shift_JIS(cp932) が多い
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_documents(files: list[tuple[str, bytes]]) -> tuple[list[Document], list[str]]:
    """(ファイル名, 中身) のリストから、PDFはページごと、テキストは1ファイルごとの Document を作る。

    戻り値: (Documentのリスト, 読み込めなかった理由などの注意のリスト)
    """
    docs: list[Document] = []
    notes: list[str] = []
    for name, data in files:
        lower = name.lower()
        if lower.endswith(".pdf"):
            try:
                with pymupdf.open(stream=data, filetype="pdf") as pdf:
                    found = 0
                    for i, page in enumerate(pdf):
                        text = page.get_text().strip()
                        if text:
                            docs.append(Document(page_content=text, metadata={"source": name, "page": i + 1}))
                            found += 1
            except Exception:
                notes.append(f"「{name}」を開けませんでした（壊れているか、パスワード付きの可能性があります）。")
                continue
            if found == 0:
                notes.append(f"「{name}」からは文字を取り出せませんでした（画像だけのPDFには対応していません）。")
        elif lower.endswith((".txt", ".md")):
            text = _decode_text(data).strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": name}))
            else:
                notes.append(f"「{name}」は空でした。")
        else:
            notes.append(f"「{name}」は対応していない形式です（PDF・txt・md に対応）。")
    return docs, notes


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    return splitter.split_documents(docs)


# ---------- 2. Chroma への登録 ----------
def build_vectorstore(chunks: list[Document], embeddings=None) -> Chroma:
    """チャンクを埋め込んで、メモリ上の Chroma に登録する。

    利用者ごとに別の collection_name を使う（同じサーバー上の他の利用者の資料と混ざらないようにするため）。
    """
    if not chunks:
        raise RagError("読み込める文章がありませんでした。")
    if len(chunks) > MAX_CHUNKS:
        raise RagError(f"資料が大きすぎます（{len(chunks)}個に分割されました。上限は{MAX_CHUNKS}個です）。ファイルを減らしてください。")
    store = Chroma(
        collection_name=f"docs_{uuid.uuid4().hex[:16]}",
        embedding_function=embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    for i in range(0, len(chunks), BATCH_SIZE):
        store.add_documents(chunks[i : i + BATCH_SIZE])
    return store


def drop_vectorstore(store: Chroma | None) -> None:
    """不要になったコレクションを消す（失敗しても無視してよい後片付け）。"""
    if store is None:
        return
    try:
        store.delete_collection()
    except Exception:
        pass


# ---------- 3. 会話履歴つきの RAG ----------
def build_rag_chain(store: Chroma, llm=None, k: int = 4):
    llm = llm or ChatOpenAI(model=CHAT_MODEL, temperature=0)
    retriever = store.as_retriever(search_kwargs={"k": k})
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        ChatPromptTemplate.from_messages(
            [("system", CONTEXTUALIZE_PROMPT), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        ),
    )
    answer_chain = create_stuff_documents_chain(
        llm,
        ChatPromptTemplate.from_messages(
            [("system", QA_PROMPT), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        ),
    )
    return create_retrieval_chain(history_aware_retriever, answer_chain)


def to_chat_history(messages: list[dict]) -> list:
    """画面の会話 [{"role": "user"/"assistant", "content": ...}] を、チェーン用のメッセージにする。

    AIの発言は AIMessage で包む（文字列のまま渡すと、人間の発言として扱われてしまう）。
    """
    history = []
    for m in messages[-HISTORY_LIMIT:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        history.append(cls(content=m["content"]))
    return history


def format_sources(context_docs: list[Document]) -> list[dict]:
    """参照された抜粋から、重複のない出典（ファイル名とページ）と冒頭の文を作る。"""
    seen: dict[str, str] = {}
    for d in context_docs:
        label = d.metadata.get("source", "不明")
        if "page" in d.metadata:
            label += f" p.{d.metadata['page']}"
        seen.setdefault(label, " ".join(d.page_content.split())[:120])
    return [{"label": label, "snippet": snippet} for label, snippet in seen.items()]


def stream_answer(chain, question: str, messages: list[dict], sources_out: list):
    """回答を文字列の断片として順に返す。参照した出典は sources_out に入れる。

    messages は「この質問より前」の会話。
    """
    for chunk in chain.stream({"input": question, "chat_history": to_chat_history(messages)}):
        if "context" in chunk:
            sources_out.extend(format_sources(chunk["context"]))
        if "answer" in chunk:
            yield chunk["answer"]
