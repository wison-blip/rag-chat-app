import os

import streamlit as st
from dotenv import load_dotenv

import rag

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

MAX_FILES = 10

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "store" not in st.session_state:
    st.session_state["store"] = None
    st.session_state["doc_info"] = None


def show_sources(sources: list[dict]):
    with st.expander("参照した資料"):
        for s in sources:
            st.write(f"**{s['label']}**")
            st.caption(f"{s['snippet']}…")


# ---------- サイドバー: 資料の読み込み ----------
with st.sidebar:
    st.header("1. 資料を読み込む")
    files = st.file_uploader(
        "PDF・テキスト（.txt / .md）を選んでください",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    k = st.slider("参照する箇所の数", min_value=2, max_value=8, value=4, help="質問のたびに、資料から探して回答の根拠にする抜粋の数です。")

    if st.button("資料を読み込む", type="primary"):
        if not files:
            st.error("ファイルを選んでください。")
        elif len(files) > MAX_FILES:
            st.error(f"ファイルは{MAX_FILES}個までです。")
        elif not os.environ.get("OPENAI_API_KEY"):
            st.error("OpenAI APIキーが設定されていません。.envファイル、またはシークレットに設定してください。")
        else:
            try:
                with st.spinner("資料を読み込んでいます..."):
                    docs, notes = rag.load_documents([(f.name, f.getvalue()) for f in files])
                    for note in notes:
                        st.warning(note)
                    chunks = rag.split_documents(docs)
                    new_store = rag.build_vectorstore(chunks)
                rag.drop_vectorstore(st.session_state["store"])  # 前の資料は捨てる
                st.session_state["store"] = new_store
                st.session_state["doc_info"] = {
                    "files": sorted({d.metadata["source"] for d in docs}),
                    "chunks": len(chunks),
                }
                st.session_state["messages"] = []  # 資料が変わったので、会話も新しくする
            except rag.RagError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"資料を読み込めませんでした（{e.__class__.__name__}）。もう一度お試しください。")

    info = st.session_state["doc_info"]
    if info:
        st.success(f"{len(info['files'])}件の資料を読み込みました（{info['chunks']}個に分割）。")
        for name in info["files"]:
            st.caption(f"・{name}")

    st.divider()
    if st.button("会話をリセット"):
        st.session_state["messages"] = []
        st.rerun()
    st.caption("アップロードした資料は、質問に答えるため OpenAI の API へ送られます。このアプリには保存せず、ブラウザを閉じると消えます。")

# ---------- メイン: チャット ----------
st.title("資料に質問できるチャット")
st.write("##### このアプリについて")
st.write("アップロードした自分の資料（PDFなど）の内容をもとに、AIが質問に答えます。回答には、根拠にした資料の場所も表示します。")
st.write("##### 使い方")
st.write(
    "1. 左のサイドバーで資料を選び、「資料を読み込む」を押します。\n"
    "2. 画面下の入力欄に質問を入力します。会話の流れを踏まえて答えるので、「それは何日まで？」のような続きの質問もできます。"
)

store = st.session_state["store"]
if store is None:
    st.info("まず、左のサイドバーから資料を読み込んでください。")

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

question = st.chat_input("資料について質問してください", disabled=store is None)
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        sources: list[dict] = []
        try:
            chain = rag.build_rag_chain(store, k=k)
            answer = st.write_stream(rag.stream_answer(chain, question, st.session_state["messages"], sources))
        except Exception as e:
            st.error(f"回答を作れませんでした（{e.__class__.__name__}）。もう一度お試しください。")
            st.stop()
        if sources:
            show_sources(sources)
    st.session_state["messages"] += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": sources},
    ]
