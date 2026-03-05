"""
Semantix — Enterprise Video Intelligence Platform
App.py — Main Streamlit application (capital A for Streamlit Cloud)
"""

import re
import time
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.video_processor import VideoProcessor, VideoMetadata, fetch_youtube_transcript, fetch_youtube_metadata
from core.scene_detector import Scene
from core.ad_engine import AdMatchingEngine, create_default_inventory
from core.search_engine import HybridSearchEngine
from core.embeddings import _IAB_NAMES

st.set_page_config(
    page_title="Semantix · Video Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stSidebar"] { background-color: #0e1016 !important; }
[data-testid="stSidebar"] > div:first-child { background-color: #0e1016 !important; }
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important; text-align: left !important;
    justify-content: flex-start !important; background: transparent !important;
    border: none !important; border-radius: 6px !important;
    color: #9ca3af !important; font-size: 0.875rem !important;
    font-weight: 500 !important; padding: 0.5rem 0.8rem !important;
    box-shadow: none !important; margin: 1px 0 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1a1d26 !important; color: #e5e7eb !important;
    transform: none !important; box-shadow: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #161820 !important; border-radius: 8px !important;
    padding: 3px !important; border: 1px solid #252830 !important; gap: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important; color: #6b7280 !important;
    border-radius: 6px !important; font-size: 0.83rem !important;
    font-weight: 500 !important; border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #f59e0b !important; color: #111827 !important; font-weight: 600 !important;
}
[data-testid="stMetric"] {
    background: #13151c !important; border: 1px solid #252830 !important;
    border-radius: 8px !important; padding: 1rem 1.1rem !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.68rem !important; font-weight: 600 !important;
    letter-spacing: 0.07em !important; text-transform: uppercase !important; color: #6b7280 !important;
}
[data-testid="stMetricValue"] > div {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important; font-weight: 500 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #252830 !important; border-radius: 8px !important; background: #13151c !important;
}
[data-testid="stProgress"] > div > div { background-color: #f59e0b !important; }
hr { border-color: #252830 !important; margin: 0.8rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# Session state
for k, v in {"videos": {}, "page": "process", "selected_video": None,
              "yt_api_key": "", "search_engine": None, "ad_engine": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.search_engine is None:
    st.session_state.search_engine = HybridSearchEngine()
if st.session_state.ad_engine is None:
    st.session_state.ad_engine = AdMatchingEngine()

PT = dict(
    plot_bgcolor="#13151c", paper_bgcolor="#13151c",
    font=dict(family="JetBrains Mono, monospace", color="#6b7280", size=11),
    margin=dict(l=16, r=16, t=44, b=16),
    colorway=["#f59e0b", "#34d399", "#60a5fa", "#a78bfa", "#fb923c", "#f472b6"],
    xaxis=dict(gridcolor="#252830", linecolor="#252830", tickfont=dict(color="#6b7280", size=10)),
    yaxis=dict(gridcolor="#252830", linecolor="#252830", tickfont=dict(color="#6b7280", size=10)),
    hoverlabel=dict(bgcolor="#1e2028", bordercolor="#363a47",
                    font=dict(family="JetBrains Mono", size=11, color="#e5e7eb")),
)

NAV = [("process","🎬","Process Video"), ("search","🔍","Semantic Search"),
       ("scenes","📺","Scene Explorer"), ("ads","📢","Ad Engine"),
       ("analytics","📊","Analytics"), ("franchise","🎯","Franchise Intel")]

with st.sidebar:
    st.markdown("## ⚡ Semantix")
    st.caption("Video Intelligence Platform")
    st.divider()
    st.caption("NAVIGATION")
    for pid, icon, label in NAV:
        active = st.session_state.page == pid
        display = f"**{icon} {label}**" if active else f"{icon} {label}"
        if st.button(display, key=f"nav_{pid}", use_container_width=True):
            st.session_state.page = pid
            st.rerun()
    st.divider()
    if st.session_state.videos:
        st.caption("ACTIVE VIDEO")
        opts = ["— All —"] + [(v.title[:28]+"…" if len(v.title)>28 else v.title)
                               for v in st.session_state.videos.values()]
        ids = [None] + list(st.session_state.videos.keys())
        sel = st.selectbox("video", opts, label_visibility="collapsed", key="vid_sel")
        st.session_state.selected_video = ids[opts.index(sel)]
        se_stats = st.session_state.search_engine.stats
        if se_stats["total_scenes"] > 0:
            c1, c2 = st.columns(2)
            c1.metric("Videos", se_stats["total_videos"])
            c2.metric("Scenes", se_stats["total_scenes"])
    st.divider()
    st.caption("SETTINGS")
    yt = st.text_input("YouTube API Key", type="password",
                        value=st.session_state.yt_api_key,
                        placeholder="Optional — for metadata", key="yt_key_in")
    if yt != st.session_state.yt_api_key:
        st.session_state.yt_api_key = yt

def _register(vm):
    st.session_state.videos[vm.video_id] = vm
    st.session_state.search_engine.add_scenes(vm.scenes)
    if st.session_state.search_engine.vectorizer is not None:
        st.session_state.ad_engine.sync_vectorizer(st.session_state.search_engine.vectorizer)
    st.session_state.selected_video = vm.video_id

def _active_vm():
    if st.session_state.selected_video:
        vm = st.session_state.videos.get(st.session_state.selected_video)
        if vm: return vm
    return next(iter(st.session_state.videos.values()), None)

def _iab_str(cats, n=3):
    return " · ".join(c["name"] for c in cats[:n]) if cats else "—"

def _sent_icon(label):
    return {"positive":"🟢","negative":"🔴","neutral":"⚪"}.get(label,"⚪")

def _yt_id(url):
    for p in [r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", r"^([A-Za-z0-9_-]{11})$"]:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def _parse_ts(t):
    t = t.strip()
    if not t: return None
    if ":" in t:
        parts = t.split(":")
        try:
            if len(parts)==2: return int(parts[0])*60+int(parts[1])
            if len(parts)==3: return int(parts[0])*3600+int(parts[1])*60+int(parts[2])
        except ValueError: return None
    try: return int(float(t))
    except ValueError: return None

def _scene_card(scene, rank=None, score=None, is_key=False):
    safety = scene.brand_safety.get("safety_score", 1.0)
    sent = scene.sentiment.get("label", "neutral")
    with st.container(border=True):
        c1, c2, c3 = st.columns([4,1,1])
        with c1:
            r = f"**#{rank}** " if rank else ""
            k = " ⭐" if is_key else ""
            st.markdown(f"{r}`{scene.start_fmt} → {scene.end_fmt}` · {scene.duration_sec:.0f}s{k}")
        c2.caption(f"🛡 {safety:.0%}")
        sc = f" `{score:.3f}`" if score is not None else ""
        c3.caption(f"{_sent_icon(sent)}{sc}")
        st.write(scene.text[:280]+("…" if len(scene.text)>280 else ""))
        st.caption(f"**{_iab_str(scene.iab_categories)}** · eng `{scene.engagement_score:.2f}` · ad suit `{scene.ad_suitability:.2f}`")
        st.progress(min(score,1.0) if score is not None else scene.ad_suitability)

def _summary(vm):
    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Scenes", vm.scene_count)
    c2.metric("Duration", vm.fmt_duration())
    c3.metric("Total Cues", vm.total_cues)
    avg = round(sum(s.duration_sec for s in vm.scenes)/max(vm.scene_count,1),1)
    c4.metric("Avg Scene", f"{avg}s")
    ca,cb = st.columns(2)
    with ca:
        st.markdown(f"**Narrative:** `{vm.narrative_structure}`")
        if vm.dominant_iab:
            st.caption("Topics: "+" · ".join(c["name"] for c in vm.dominant_iab[:5]))
    with cb:
        if vm.emotional_arc:
            df = pd.DataFrame(vm.emotional_arc)
            fig = px.area(df, x="start_sec", y="sentiment_score", color_discrete_sequence=["#f59e0b"])
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.1)")
            fig.update_layout(**PT, height=160, showlegend=False,
                              title=dict(text="Emotional Arc",font=dict(size=11,color="#6b7280")),
                              xaxis_title="",yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

def page_process():
    st.markdown("## 🎬 Process Video")
    st.caption("Upload subtitle files (SRT/VTT) or fetch directly from YouTube")
    st.divider()
    tab1,tab2,tab3 = st.tabs(["📁  Upload File","▶️  YouTube URL","📋  Paste Text"])
    with tab1:
        ca,cb = st.columns([3,2],gap="large")
        with ca:
            uploaded = st.file_uploader("Subtitle file",type=["srt","vtt"],label_visibility="collapsed")
            title1 = st.text_input("Title (optional)",placeholder="Leave blank to use filename",key="tit1")
        with cb:
            st.caption("DETECTION SETTINGS")
            min1=st.slider("Min scene (s)",10,60,20,key="min1")
            max1=st.slider("Max scene (s)",60,300,120,key="max1")
            thr1=st.slider("Sensitivity",0.2,0.7,0.35,0.05,key="thr1")
        if uploaded and st.button("⚡ Process File",key="proc1"):
            content=uploaded.read().decode("utf-8",errors="replace")
            fmt="vtt" if uploaded.name.lower().endswith(".vtt") else "srt"
            title=title1 or uploaded.name
            with st.spinner("Analysing scenes…"):
                t0=time.time()
                vm=VideoProcessor(min1,max1,thr1).process_file(content,title,fmt)
                elapsed=time.time()-t0
            if not vm.scenes: st.error("No scenes detected."); return
            _register(vm)
            st.success(f"✅ {len(vm.scenes)} scenes in {elapsed:.2f}s")
            _summary(vm)
    with tab2:
        ca,cb = st.columns([3,2],gap="large")
        with ca:
            yt_url=st.text_input("YouTube URL or ID",placeholder="https://youtube.com/watch?v=...",key="yt_url")
        with cb:
            st.caption("DETECTION SETTINGS")
            min2=st.slider("Min scene (s)",10,60,20,key="min2")
            max2=st.slider("Max scene (s)",60,300,120,key="max2")
            thr2=st.slider("Sensitivity",0.2,0.7,0.35,0.05,key="thr2")
        if yt_url and st.button("⚡ Fetch & Process",key="proc2"):
            vid_id=_yt_id(yt_url)
            if not vid_id: st.error("Could not parse video ID."); return
            with st.spinner("Fetching transcript…"):
                transcript=fetch_youtube_transcript(vid_id)
            if transcript is None: st.error("No captions available."); return
            meta=None
            if st.session_state.yt_api_key:
                with st.spinner("Fetching metadata…"):
                    meta=fetch_youtube_metadata(vid_id,st.session_state.yt_api_key)
            title=meta.get("title",f"YouTube: {vid_id}") if meta else f"YouTube: {vid_id}"
            with st.spinner("Detecting scenes…"):
                t0=time.time()
                vm=VideoProcessor(min2,max2,thr2).process_youtube_transcript(transcript,vid_id,title,meta)
                elapsed=time.time()-t0
            _register(vm)
            st.success(f"✅ {len(vm.scenes)} scenes in {elapsed:.2f}s")
            _summary(vm)
    with tab3:
        ca,cb = st.columns([3,2],gap="large")
        with ca:
            title3=st.text_input("Title",placeholder="My Video",key="tit3")
            pasted=st.text_area("Paste SRT or VTT content",height=200,
                                placeholder="1\n00:00:01,000 --> 00:00:05,000\nHello world...",key="paste3")
        with cb:
            st.caption("DETECTION SETTINGS")
            min3=st.slider("Min scene (s)",10,60,20,key="min3")
            max3=st.slider("Max scene (s)",60,300,120,key="max3")
        if pasted and st.button("⚡ Process Text",key="proc3"):
            with st.spinner("Processing…"):
                t0=time.time()
                vm=VideoProcessor(min3,max3).process_file(pasted,title3 or "Pasted Video")
                elapsed=time.time()-t0
            if not vm.scenes: st.error("No scenes found."); return
            _register(vm)
            st.success(f"✅ {len(vm.scenes)} scenes in {elapsed:.2f}s")
            _summary(vm)

def page_search():
    st.markdown("## 🔍 Semantic Search")
    st.caption("Find any moment, topic, emotion or keyword across all indexed videos")
    st.divider()
    se=st.session_state.search_engine
    if se.stats["total_scenes"]==0:
        st.info("No videos indexed yet — process a video first."); return
    mode=st.radio("Mode",["🧠 Semantic","🏷️ Tags / IAB","⏱️ By Timestamp"],
                  horizontal=True,label_visibility="collapsed")
    st.divider()
    if mode=="🧠 Semantic": _semantic_search(se)
    elif mode=="🏷️ Tags / IAB": _tag_search()
    else: _timestamp_search()

def _semantic_search(se):
    c1,c2,c3=st.columns([4,1,1])
    with c1:
        query=st.text_input("Query",placeholder="e.g. emotional confrontation / product review / landscape",
                             label_visibility="collapsed",key="sem_q")
    with c2:
        top_k=st.selectbox("Results",[5,10,20,50],index=1,label_visibility="collapsed",key="sem_k")
    with c3:
        s_opt=st.selectbox("Safety",["Any","Safe 0.5+","Brand Safe 0.8+"],
                            label_visibility="collapsed",key="sem_safe")
        smap={"Any":0.0,"Safe 0.5+":0.5,"Brand Safe 0.8+":0.8}
    ca,cb=st.columns([3,1])
    with ca:
        iab_sel=st.multiselect("IAB",[f"{k}: {v}" for k,v in _IAB_NAMES.items()],
                                placeholder="Filter by content category…",
                                label_visibility="collapsed",key="sem_iab")
        iab_filter=[s.split(":")[0] for s in iab_sel] or None
    with cb:
        diversify=st.checkbox("Diversify",value=True,key="sem_div")
    vid_filter=None
    if len(st.session_state.videos)>1:
        opts2=["All Videos"]+[vm.title for vm in st.session_state.videos.values()]
        ids2=[None]+list(st.session_state.videos.keys())
        vsel=st.selectbox("Video filter",opts2,label_visibility="collapsed",key="sem_vid")
        vid_filter=ids2[opts2.index(vsel)]
    if not query:
        st.caption("**SUGGESTED QUERIES**")
        sugs=["exciting action","emotional dialogue","product review","travel destination",
              "health tips","financial advice","comedy moment","suspenseful scene",
              "interview","tutorial","celebration","conflict"]
        cols=st.columns(4)
        for i,s in enumerate(sugs):
            if cols[i%4].button(s,key=f"sug_{i}"):
                st.session_state["sem_q"]=s; st.rerun()
        return
    with st.spinner(f"Searching…"):
        results=se.search(query,top_k=top_k,diversify=diversify,
                          video_id=vid_filter,min_safety=smap[s_opt],
                          iab_filter=iab_filter,expand=True)
    if not results: st.warning("No results. Try broader terms."); return
    st.markdown(f"**{len(results)} scenes** matched · `{query}`")
    st.divider()
    for r in results:
        scene=r.scene
        vm=st.session_state.videos.get(scene.video_id)
        is_key=vm and scene.scene_id in vm.key_scenes if vm else False
        vtitle=vm.title if vm else scene.video_id
        with st.container(border=True):
            h1,h2=st.columns([5,1])
            with h1:
                st.markdown(f"**#{r.rank}** — {vtitle}{'  ⭐' if is_key else ''}")
                st.caption(f"`{scene.start_fmt} → {scene.end_fmt}` · {scene.duration_sec:.0f}s")
            h2.metric("Score",f"{r.score:.3f}")
            st.write(scene.text[:320]+("…" if len(scene.text)>320 else ""))
            ca,cb,cc,cd=st.columns(4)
            safety=scene.brand_safety.get("safety_score",1.0)
            sent=scene.sentiment.get("label","neutral")
            ca.caption(f"**{_iab_str(scene.iab_categories,2)}**")
            cb.caption(f"{_sent_icon(sent)} {sent.title()}")
            cc.caption(f"🛡 {safety:.0%}")
            cd.caption(f"vec `{r.vector_score:.2f}` bm25 `{r.bm25_score:.2f}`")
            st.progress(min(r.score,1.0))

def _tag_search():
    st.markdown("#### 🏷️ Browse by Tags & Metadata")
    ca,cb,cc=st.columns(3)
    with ca:
        iab_f=st.multiselect("IAB Category",[f"{k}: {v}" for k,v in _IAB_NAMES.items()],
                              placeholder="Any…",key="tag_iab")
    with cb:
        sent_f=st.multiselect("Sentiment",["positive","neutral","negative"],
                               default=["positive","neutral","negative"],key="tag_sent")
    with cc:
        min_suit=st.slider("Min ad suitability",0.0,1.0,0.0,0.05,key="tag_suit")
    keyword=st.text_input("Keyword in text",placeholder="e.g. cricket / revenue / love",key="tag_kw")
    iab_codes=[x.split(":")[0] for x in iab_f]
    filtered=[]
    for vm in st.session_state.videos.values():
        for s in vm.scenes:
            if s.sentiment.get("label","neutral") not in sent_f: continue
            if s.ad_suitability<min_suit: continue
            if iab_codes:
                scene_iab=[c.get("iab_code",c.get("id","")) for c in s.iab_categories]
                if not any(code in scene_iab for code in iab_codes): continue
            if keyword and keyword.lower() not in s.text.lower(): continue
            filtered.append((vm,s))
    st.caption(f"**{len(filtered)}** scenes match")
    for vm,scene in filtered[:50]:
        _scene_card(scene,is_key=scene.scene_id in vm.key_scenes)

def _timestamp_search():
    st.markdown("#### ⏱️ Browse by Timestamp")
    if not st.session_state.videos: st.info("No videos loaded."); return
    vm=_active_vm()
    if not vm: return
    st.caption(f"**{vm.title}** · {vm.fmt_duration()} · {vm.scene_count} scenes")
    ca,cb=st.columns(2)
    with ca: ts_start=st.text_input("From",placeholder="00:01:30 or 90s",key="ts_from")
    with cb: ts_end=st.text_input("To (optional)",placeholder="00:05:00 or 300s",key="ts_to")
    start_s=_parse_ts(ts_start) if ts_start else None
    end_s=_parse_ts(ts_end) if ts_end else None
    st.divider()
    shown=0
    for scene in vm.scenes:
        if start_s is not None and scene.end_sec<start_s: continue
        if end_s is not None and scene.start_sec>end_s: continue
        _scene_card(scene,is_key=scene.scene_id in vm.key_scenes)
        shown+=1
        if shown>=30: st.caption(f"Showing 30 of {vm.scene_count} scenes"); break

def page_scenes():
    st.markdown("## 📺 Scene Explorer")
    st.caption("Deep-dive into every detected scene with full metadata")
    st.divider()
    if not st.session_state.videos: st.info("No videos processed yet."); return
    vm=_active_vm()
    if not vm: return
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Scenes",vm.scene_count)
    c2.metric("Duration",vm.fmt_duration())
    avg_eng=round(sum(s.engagement_score for s in vm.scenes)/max(vm.scene_count,1),3)
    c3.metric("Avg Engagement",avg_eng)
    avg_safe=sum(s.brand_safety.get("safety_score",1.0) for s in vm.scenes)/max(vm.scene_count,1)
    c4.metric("Avg Safety",f"{avg_safe:.0%}")
    c5.metric("Narrative",vm.narrative_structure.split("(")[0].strip()[:14])
    st.divider()
    tab1,tab2,tab3=st.tabs(["🎭  Emotional Arc","📊  Analysis","🗂  Scene List"])
    with tab1:
        if vm.emotional_arc:
            df=pd.DataFrame(vm.emotional_arc)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=df["start_sec"],y=df["sentiment_score"],fill="tozeroy",
                fillcolor="rgba(245,158,11,0.08)",line=dict(color="#f59e0b",width=2),
                mode="lines+markers",marker=dict(size=5),name="Sentiment"))
            fig.add_trace(go.Scatter(x=df["start_sec"],y=df["engagement"],
                line=dict(color="#34d399",width=2,dash="dot"),mode="lines",name="Engagement"))
            fig.add_hline(y=0,line_dash="dot",line_color="rgba(255,255,255,0.08)")
            for kid in vm.key_scenes:
                ks=next((s for s in vm.scenes if s.scene_id==kid),None)
                if ks: fig.add_vline(x=ks.start_sec,line_dash="dash",line_color="rgba(245,158,11,0.3)")
            fig.update_layout(**PT,height=340,
                title=dict(text="Sentiment & Engagement · dashed lines = key scenes",font=dict(size=12,color="#6b7280")),
                xaxis_title="Time (s)",yaxis_title="Score",
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#6b7280")))
            st.plotly_chart(fig,use_container_width=True)
    with tab2:
        ca,cb=st.columns(2)
        with ca:
            iab_c={}
            for s in vm.scenes:
                for cat in s.iab_categories[:1]: iab_c[cat["name"]]=iab_c.get(cat["name"],0)+1
            if iab_c:
                df2=pd.DataFrame(sorted(iab_c.items(),key=lambda x:x[1],reverse=True)[:8],columns=["Category","Scenes"])
                fig=px.bar(df2,x="Scenes",y="Category",orientation="h",color_discrete_sequence=["#f59e0b"])
                fig.update_layout(**PT,height=300,title=dict(text="Top IAB Categories",font=dict(size=12,color="#6b7280")),
                                  yaxis=dict(autorange="reversed",**PT["yaxis"]))
                st.plotly_chart(fig,use_container_width=True)
        with cb:
            fig=px.histogram([s.brand_safety.get("safety_score",1.0) for s in vm.scenes],
                             nbins=10,color_discrete_sequence=["#34d399"])
            fig.update_layout(**PT,height=300,showlegend=False,
                title=dict(text="Brand Safety Distribution",font=dict(size=12,color="#6b7280")),
                xaxis_title="Safety Score",yaxis_title="Count")
            st.plotly_chart(fig,use_container_width=True)
        eng_s=[s.engagement_score for s in vm.scenes]
        suit_s=[s.ad_suitability for s in vm.scenes]
        times=[s.start_sec for s in vm.scenes]
        labels=[s.start_fmt for s in vm.scenes]
        fig=go.Figure(go.Scatter(x=times,y=eng_s,mode="markers+text",
            text=labels,textposition="top center",textfont=dict(size=8,color="#6b7280"),
            marker=dict(size=[s*18+6 for s in suit_s],color=eng_s,
                        colorscale=[[0,"#1e2028"],[1,"#f59e0b"]],showscale=True,opacity=0.8,
                        colorbar=dict(title="Engagement",tickfont=dict(color="#6b7280"))),
            hovertemplate="%{text}  eng: %{y:.3f}<extra></extra>"))
        fig.update_layout(**PT,height=300,title=dict(text="Scene Map — bubble = ad suitability",
            font=dict(size=12,color="#6b7280")),xaxis_title="Time (s)",yaxis_title="Engagement")
        st.plotly_chart(fig,use_container_width=True)
    with tab3:
        fc1,fc2,fc3=st.columns(3)
        with fc1: sent_f=st.multiselect("Sentiment",["positive","neutral","negative"],
                                         default=["positive","neutral","negative"],key="sc_sent")
        with fc2: min_eng=st.slider("Min engagement",0.0,1.0,0.0,0.05,key="sc_eng")
        with fc3: min_safe2=st.slider("Min safety",0.0,1.0,0.0,0.05,key="sc_safe")
        filtered=[s for s in vm.scenes
                  if s.sentiment.get("label","neutral") in sent_f
                  and s.engagement_score>=min_eng
                  and s.brand_safety.get("safety_score",1.0)>=min_safe2]
        st.caption(f"**{len(filtered)}** of {vm.scene_count} scenes")
        for s in filtered: _scene_card(s,is_key=s.scene_id in vm.key_scenes)

def page_ads():
    st.markdown("## 📢 Ad Engine")
    st.caption("Contextual ad matching and placement optimisation")
    st.divider()
    if not st.session_state.videos: st.info("Process a video first."); return
    vm=_active_vm()
    if not vm: return
    ae=st.session_state.ad_engine
    tab1,tab2,tab3=st.tabs(["🎯  Placement Plan","🔍  Scene Matching","📦  Inventory"])
    with tab1:
        ca,cb=st.columns([2,1],gap="large")
        with ca:
            p_types=st.multiselect("Placement types",["pre-roll","mid-roll","post-roll"],
                                   default=["pre-roll","mid-roll","post-roll"])
            min_safe_ad=st.slider("Min brand safety",0.0,1.0,0.5,0.05,key="ad_safe")
        with cb:
            st.metric("Scenes",vm.scene_count); st.metric("Duration",vm.fmt_duration())
        if st.button("⚡ Generate Placement Plan",key="gen_plan"):
            with st.spinner("Optimising…"):
                placements=ae.plan_placements(vm.scenes,vm.duration_ms,p_types)
                perf=ae.simulate_performance(placements)
                st.session_state["_pl"]=placements; st.session_state["_perf"]=perf
        if "_pl" in st.session_state and st.session_state["_pl"]:
            pl=st.session_state["_pl"]; perf=st.session_state["_perf"]
            st.divider()
            c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("Placements",perf["total_placements"])
            c2.metric("Est. Revenue",f"${perf['total_revenue_usd']:.2f}")
            c3.metric("Impressions",f"{perf['total_impressions']:,}")
            c4.metric("Clicks",f"{perf['estimated_clicks']:,}")
            c5.metric("Avg CPM",f"${perf['avg_cpm']:.2f}")
            p_df=pd.DataFrame([p.to_dict() for p in pl])
            colors={"pre-roll":"#f59e0b","mid-roll":"#34d399","post-roll":"#60a5fa"}
            fig=go.Figure()
            for pt in p_df["placement_type"].unique():
                sub=p_df[p_df["placement_type"]==pt]
                fig.add_trace(go.Scatter(x=sub["timestamp_ms"]/1000,y=sub["relevance_score"],
                    mode="markers+text",name=pt,text=sub["brand"].str[:12],
                    textposition="top center",textfont=dict(size=9,color="#9ca3af"),
                    marker=dict(size=sub["estimated_cpm"]*2.5+8,color=colors.get(pt,"#f59e0b"),opacity=0.85)))
            fig.update_layout(**PT,height=320,
                title=dict(text="Placement Timeline",font=dict(size=12,color="#6b7280")),
                xaxis_title="Time (s)",yaxis_title="Relevance",
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#6b7280")))
            st.plotly_chart(fig,use_container_width=True)
            show=["timestamp_fmt","placement_type","ad_title","brand","relevance_score","safety_score","estimated_cpm"]
            st.dataframe(p_df[show].rename(columns={"timestamp_fmt":"Time","placement_type":"Type",
                "ad_title":"Ad","brand":"Brand","relevance_score":"Relevance",
                "safety_score":"Safety","estimated_cpm":"CPM ($)"}),
                use_container_width=True,hide_index=True)
    with tab2:
        opts=[f"[{s.start_fmt}] {s.text[:60]}…" for s in vm.scenes]
        idx=st.selectbox("Select scene",range(len(vm.scenes)),format_func=lambda i:opts[i],key="ad_scene")
        scene=vm.scenes[idx]; matches=ae.match_ads(scene,top_k=5)
        if not matches: st.warning("No eligible ads.")
        else:
            for ad,si in matches:
                with st.container(border=True):
                    h1,h2=st.columns([4,1])
                    with h1: st.markdown(f"**{ad.title}** · {ad.brand}"); st.caption(ad.description)
                    h2.metric("Match",f"{si['total']:.3f}")
                    ca,cb,cc,cd,ce=st.columns(5)
                    ca.caption(f"content `{si['content_sim']:.2f}`"); cb.caption(f"IAB `{si['iab_match']:.2f}`")
                    cc.caption(f"safety `{si['safety']:.2f}`"); cd.caption(f"demo `{si['demographic']:.2f}`")
                    ce.caption(f"perf `{si['performance']:.2f}`"); st.progress(min(si["total"],1.0))
    with tab3:
        inv_df=pd.DataFrame([ad.to_dict() for ad in ae.inventory])
        cols=["title","brand","cpm_base","historical_ctr","performance_score","brand_safety_min","budget_remaining"]
        st.dataframe(inv_df[cols].rename(columns={"title":"Ad","brand":"Brand","cpm_base":"Base CPM",
            "historical_ctr":"CTR","performance_score":"Perf","brand_safety_min":"Min Safety","budget_remaining":"Budget Left"}),
            use_container_width=True,hide_index=True)

def page_analytics():
    st.markdown("## 📊 Analytics Dashboard")
    st.caption("Content intelligence metrics across all indexed videos")
    st.divider()
    if not st.session_state.videos: st.info("Process a video to see analytics."); return
    all_s=[s for vm in st.session_state.videos.values() for s in vm.scenes]
    n=max(len(all_s),1)
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Videos",len(st.session_state.videos)); c2.metric("Total Scenes",len(all_s))
    c3.metric("Avg Engagement",round(sum(s.engagement_score for s in all_s)/n,3))
    c4.metric("Brand Safe",f"{sum(1 for s in all_s if s.brand_safety.get('safety_score',1.0)>=0.8)/n:.0%}")
    c5.metric("Positive Sent.",f"{sum(1 for s in all_s if s.sentiment.get('label')=='positive')/n:.0%}")
    st.divider()
    ca,cb=st.columns(2)
    with ca:
        labels=[s.sentiment.get("label","neutral") for s in all_s]
        lc={l:labels.count(l) for l in set(labels)}
        fig=px.pie(values=list(lc.values()),names=list(lc.keys()),hole=0.52,
                   color_discrete_map={"positive":"#34d399","neutral":"#4b5563","negative":"#f87171"})
        fig.update_layout(**PT,height=280,title=dict(text="Sentiment Breakdown",font=dict(size=12,color="#6b7280")),
                          legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#9ca3af")))
        st.plotly_chart(fig,use_container_width=True)
    with cb:
        iab_all={}
        for s in all_s:
            for cat in s.iab_categories[:1]: iab_all[cat["name"]]=iab_all.get(cat["name"],0)+1
        if iab_all:
            top=sorted(iab_all.items(),key=lambda x:x[1],reverse=True)[:10]
            df2=pd.DataFrame(top,columns=["Category","Scenes"])
            fig=px.bar(df2,x="Scenes",y="Category",orientation="h",color="Scenes",
                       color_continuous_scale=[[0,"#252830"],[1,"#f59e0b"]])
            fig.update_layout(**PT,height=280,coloraxis_showscale=False,
                title=dict(text="Top Content Categories",font=dict(size=12,color="#6b7280")),
                yaxis=dict(autorange="reversed",**PT["yaxis"]))
            st.plotly_chart(fig,use_container_width=True)
    ca,cb=st.columns(2)
    with ca:
        eng_v=[s.engagement_score for s in all_s]; safe_v=[s.brand_safety.get("safety_score",1.0) for s in all_s]
        suit_v=[s.ad_suitability for s in all_s]
        fig=go.Figure(go.Scatter(x=eng_v,y=safe_v,mode="markers",
            marker=dict(size=[v*14+5 for v in suit_v],color=suit_v,
                        colorscale=[[0,"#1e2028"],[1,"#f59e0b"]],showscale=True,opacity=0.75,
                        colorbar=dict(title="Ad Suit.",tickfont=dict(color="#6b7280"))),
            hovertemplate="eng: %{x:.2f} · safety: %{y:.2f}<extra></extra>"))
        fig.update_layout(**PT,height=300,
            title=dict(text="Engagement vs Safety",font=dict(size=12,color="#6b7280")),
            xaxis_title="Engagement",yaxis_title="Safety")
        st.plotly_chart(fig,use_container_width=True)
    with cb:
        fig=px.histogram([s.ad_suitability for s in all_s],nbins=15,color_discrete_sequence=["#fb923c"])
        fig.update_layout(**PT,height=300,showlegend=False,
            title=dict(text="Ad Suitability Distribution",font=dict(size=12,color="#6b7280")),
            xaxis_title="Ad Suitability",yaxis_title="Scenes")
        st.plotly_chart(fig,use_container_width=True)
    st.divider(); st.caption("VIDEO BREAKDOWN")
    rows=[]
    for vm in st.session_state.videos.values():
        if not vm.scenes: continue
        rows.append({"Title":vm.title[:40],"Duration":vm.fmt_duration(),"Scenes":vm.scene_count,
            "Narrative":vm.narrative_structure.split("(")[0].strip(),
            "Avg Engagement":round(sum(s.engagement_score for s in vm.scenes)/len(vm.scenes),3),
            "Brand Safe %":f"{sum(s.brand_safety.get('safety_score',1)>=0.8 for s in vm.scenes)/len(vm.scenes):.0%}",
            "Top Topic":vm.dominant_iab[0]["name"] if vm.dominant_iab else "—"})
    if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

def page_franchise():
    st.markdown("## 🎯 Franchise Intelligence")
    st.caption("Cross-video theme tracking and recurring ad opportunities")
    st.divider()
    if not st.session_state.videos: st.info("Process at least one video first."); return
    all_scenes=[s for vm in st.session_state.videos.values() for s in vm.scenes]
    se=st.session_state.search_engine
    theme_data=[]
    for vm in st.session_state.videos.values():
        for theme in vm.franchise_themes:
            theme_data.append({"Theme":theme,"Video":vm.title[:25]})
    if theme_data:
        df_t=pd.DataFrame(theme_data)
        df_count=df_t.groupby(["Theme","Video"]).size().reset_index(name="Count")
        fig=px.bar(df_count,x="Theme",y="Count",color="Video",color_discrete_sequence=PT["colorway"])
        fig.update_layout(**PT,height=300,title=dict(text="Recurring Themes by Video",font=dict(size=12,color="#6b7280")),
                          xaxis_tickangle=-30,legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#9ca3af")))
        st.plotly_chart(fig,use_container_width=True)
    st.divider(); st.caption("TOP AD OPPORTUNITIES")
    iab_opps={}
    for vm in st.session_state.videos.values():
        for s in vm.scenes:
            if s.ad_suitability>0.55:
                for cat in s.iab_categories[:1]:
                    iab_opps.setdefault(cat["name"],[]).append({"video":vm.title[:28],"suitability":s.ad_suitability,"engagement":s.engagement_score})
    if iab_opps:
        sorted_opps=sorted(iab_opps.items(),key=lambda x:sum(o["suitability"] for o in x[1])/len(x[1]),reverse=True)
        for cat_name,ops in sorted_opps[:6]:
            avg_suit=sum(o["suitability"] for o in ops)/len(ops)
            avg_eng=sum(o["engagement"] for o in ops)/len(ops)
            n_vids=len(set(o["video"] for o in ops))
            with st.container(border=True):
                ca,cb,cc,cd=st.columns([3,1,1,1])
                ca.markdown(f"**{cat_name}**"); ca.caption(f"{len(ops)} scenes · {n_vids} video(s)")
                cb.metric("Avg Suit.",f"{avg_suit:.3f}"); cc.metric("Avg Eng.",f"{avg_eng:.3f}"); cd.metric("Videos",n_vids)
                st.progress(avg_suit)
    if len(all_scenes)>1 and se.stats["total_scenes"]>0:
        st.divider(); st.caption("SIMILAR SCENES ACROSS VIDEOS")
        preview=all_scenes[:60]
        opts=[f"[{s.video_id[:8]}] {s.start_fmt} — {s.text[:50]}…" for s in preview]
        sel=st.selectbox("Reference scene",range(len(preview)),format_func=lambda i:opts[i],key="fr_sim")
        similar=se.find_similar_scenes(preview[sel],top_k=6,exclude_same_video=True)
        if similar:
            for r in similar:
                vm2=st.session_state.videos.get(r.scene.video_id)
                with st.container(border=True):
                    ca,cb=st.columns([5,1])
                    ca.markdown(f"**{vm2.title[:35] if vm2 else r.scene.video_id}**")
                    ca.caption(f"`{r.scene.start_fmt} → {r.scene.end_fmt}`")
                    ca.write(r.scene.text[:200]+"…"); cb.metric("Sim.",f"{r.score:.3f}")
                    st.progress(min(r.score,1.0))
        else: st.info("No similar scenes in other videos.")

{
    "process":   page_process,
    "search":    page_search,
    "scenes":    page_scenes,
    "ads":       page_ads,
    "analytics": page_analytics,
    "franchise": page_franchise,
}[st.session_state.page]()
