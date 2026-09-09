from __future__ import annotations
import io, json, zipfile, re
import numpy as np
import pandas as pd
import streamlit as st

# Landmarks mínimos que ya están presentes en la secuencia V104/V107 y son útiles
# para el ajuste articular inicial de SKEL.
JOINTS = [
    "LHip","RHip","LKnee","RKnee","LAnkle","RAnkle",
    "LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist",
    "Neck","Head","Hip"
]

# Alias tolerantes para no depender de una única convención de nombres.
_ALIASES = {
    "lhip": "LHip", "lefthip": "LHip", "left_hip": "LHip", "hipleft": "LHip",
    "rhip": "RHip", "righthip": "RHip", "right_hip": "RHip", "hipright": "RHip",
    "lknee": "LKnee", "leftknee": "LKnee", "left_knee": "LKnee", "kneeleft": "LKnee",
    "rknee": "RKnee", "rightknee": "RKnee", "right_knee": "RKnee", "kneeright": "RKnee",
    "lankle": "LAnkle", "leftankle": "LAnkle", "left_ankle": "LAnkle", "ankleleft": "LAnkle",
    "rankle": "RAnkle", "rightankle": "RAnkle", "right_ankle": "RAnkle", "ankleright": "RAnkle",
    "lshoulder": "LShoulder", "leftshoulder": "LShoulder", "left_shoulder": "LShoulder",
    "rshoulder": "RShoulder", "rightshoulder": "RShoulder", "right_shoulder": "RShoulder",
    "lelbow": "LElbow", "leftelbow": "LElbow", "left_elbow": "LElbow",
    "relbow": "RElbow", "rightelbow": "RElbow", "right_elbow": "RElbow",
    "lwrist": "LWrist", "leftwrist": "LWrist", "left_wrist": "LWrist",
    "rwrist": "RWrist", "rightwrist": "RWrist", "right_wrist": "RWrist",
    "neck": "Neck", "head": "Head", "nose": "Nose", "hip": "Hip",
}

def _norm_name(name: str) -> str:
    s = str(name).strip()
    compact = re.sub(r"[^a-z0-9]", "", s.lower())
    # Primero coincidencia exacta canónica.
    for canon in JOINTS + ["Nose"]:
        if compact == re.sub(r"[^a-z0-9]", "", canon.lower()):
            return canon
    # Después alias con y sin separadores.
    raw = s.lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(raw, _ALIASES.get(compact, s))

def _xyz(v):
    try:
        if isinstance(v, dict):
            # Aceptar X/Y/Z y x/y/z.
            keys = {str(k).lower(): k for k in v.keys()}
            if all(k in keys for k in ("x","y","z")):
                a = [v[keys["x"]], v[keys["y"]], v[keys["z"]]]
            else:
                return None
        elif isinstance(v, (list, tuple, np.ndarray, pd.Series)) and len(v) >= 3:
            a = [v[0], v[1], v[2]]
        else:
            return None
        a = [float(x) for x in a]
        return a if np.isfinite(a).all() else None
    except Exception:
        return None

def _frame_points_from_frame(f):
    if not isinstance(f, dict):
        return {}
    # V104/V107 oficial usa `joints`; se mantienen los otros nombres como compatibilidad.
    src = f.get("joints")
    if not isinstance(src, dict) or not src:
        src = f.get("points")
    if not isinstance(src, dict) or not src:
        src = f.get("landmarks")
    if not isinstance(src, dict):
        return {}
    out = {}
    for k, v in src.items():
        p = _xyz(v)
        if p is not None:
            out[_norm_name(k)] = p

    # Centros derivados sólo para visualización/registro inicial. No sustituyen landmarks medidos.
    if "Hip" not in out and "LHip" in out and "RHip" in out:
        out["Hip"] = ((np.asarray(out["LHip"]) + np.asarray(out["RHip"])) / 2.0).tolist()
    if "Neck" not in out and "LShoulder" in out and "RShoulder" in out:
        out["Neck"] = ((np.asarray(out["LShoulder"]) + np.asarray(out["RShoulder"])) / 2.0).tolist()
    if "Head" not in out and "Nose" in out:
        out["Head"] = list(out["Nose"])
    return out

def _select_best_frame(motion):
    frames = list((motion or {}).get("frames") or [])
    best_i, best_pts, best_score = 0, {}, -1
    for i, f in enumerate(frames):
        pts = _frame_points_from_frame(f)
        score = sum(1 for j in JOINTS if j in pts)
        if score > best_score:
            best_i, best_pts, best_score = i, pts, score
        # 14+ ya es un frame excelente para este PoC; evitamos recorrer de más.
        if score >= 14:
            break
    return best_i, best_pts, max(0, best_score)

def _target_csv(points, frame_index=0):
    rows = []
    for j in JOINTS:
        if j in points:
            x, y, z = points[j]
            rows.append({"frame": int(frame_index)+1, "joint": j, "x": x, "y": y, "z": z,
                         "source": "derived" if j in ("Hip",) else "V104/V107"})
    return pd.DataFrame(rows)

def _inspect_private_bundle(data: bytes):
    info={"has_male":False,"has_female":False,"files":[],"valid_zip":False}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names=z.namelist(); info["valid_zip"]=True
            info["files"]=names[:80]
            low=[n.lower() for n in names]
            info["has_male"]=any(n.endswith("skel_male.pkl") for n in low)
            info["has_female"]=any(n.endswith("skel_female.pkl") for n in low)
    except Exception as e:
        info["error"]=str(e)
    return info

def _plot_frame(df):
    if df.empty:
        return
    try:
        import plotly.graph_objects as go
        links = [
            ("LShoulder","RShoulder"),("LShoulder","LElbow"),("LElbow","LWrist"),
            ("RShoulder","RElbow"),("RElbow","RWrist"),("LShoulder","LHip"),
            ("RShoulder","RHip"),("LHip","RHip"),("LHip","LKnee"),("LKnee","LAnkle"),
            ("RHip","RKnee"),("RKnee","RAnkle"),("Neck","LShoulder"),("Neck","RShoulder"),("Neck","Head")
        ]
        P={r.joint:(r.x,r.y,r.z) for r in df.itertuples()}
        fig=go.Figure()
        for a,b in links:
            if a in P and b in P:
                xa,ya,za=P[a]; xb,yb,zb=P[b]
                fig.add_trace(go.Scatter3d(x=[xa,xb],y=[ya,yb],z=[za,zb],mode="lines",showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=df.x,y=df.y,z=df.z,mode="markers+text",text=df.joint,
                                   textposition="top center",name="Landmarks objetivo"))
        fig.update_layout(height=520,margin=dict(l=0,r=0,t=35,b=0),title="Frame objetivo V110 · XYZ para ajuste SKEL",
                          scene=dict(aspectmode="data"))
        st.plotly_chart(fig,use_container_width=True)
    except Exception as exc:
        st.caption(f"Visualización 3D no disponible: {exc}")

def render_skel_poc_panel(motion):
    frames=list((motion or {}).get("frames") or [])
    if not frames:
        st.warning("No hay secuencia V104/V107 disponible para construir el frame objetivo de SKEL.")
        return

    best_i, pts, score = _select_best_frame(motion)
    df = _target_csv(pts, best_i)
    raw_count = len(_frame_points_from_frame(frames[best_i])) if frames else 0

    st.success(f"Motor cinemático disponible: {len(frames)} frames. V110 usa el primer frame con cobertura articular suficiente como puerta de validación antes de animar SKEL.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Frames V104/V107",len(frames))
    c2.metric("Landmarks objetivo",len(df))
    c3.metric("Frame seleccionado",f"{best_i+1}/{len(frames)}")
    c4.metric("SKEL","entrada XYZ")

    if len(df) == 0:
        st.error("V109.1 sigue sin encontrar landmarks compatibles dentro del payload V104/V107.")
        st.write("Claves presentes en el frame seleccionado:", list((frames[best_i].get("joints") or {}).keys())[:40])
        return
    elif len(df) < 10:
        st.warning(f"Sólo se han recuperado {len(df)} landmarks objetivo. El CSV es utilizable para diagnóstico, pero todavía no para un ajuste SKEL fiable.")
    else:
        st.info(f"Puente V104/V107 → SKEL recuperado: {len(df)} landmarks objetivo de {raw_count} puntos XYZ disponibles en el frame {best_i+1}.")

    st.download_button("⬇️ Frame objetivo V110 (CSV)",df.to_csv(index=False).encode("utf-8-sig"),
                       "V110_SKEL_target_frame.csv","text/csv",use_container_width=True)
    st.caption("Este CSV contiene las coordenadas XYZ que se usarán para el ajuste articular. Los centros derivados están identificados y no modifican ninguna métrica clínica.")

    _plot_frame(df)
    with st.expander("Ver coordenadas XYZ del frame objetivo", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Diagnóstico geométrico previo al fitting: no altera datos ni métricas clínicas.
    P={r.joint:np.array([r.x,r.y,r.z],dtype=float) for r in df.itertuples()}
    def dist(a,b):
        return float(np.linalg.norm(P[a]-P[b])) if a in P and b in P else float("nan")
    segs={
        "Pelvis L-R":dist("LHip","RHip"),
        "Fémur L":dist("LHip","LKnee"), "Fémur R":dist("RHip","RKnee"),
        "Tibia L":dist("LKnee","LAnkle"), "Tibia R":dist("RKnee","RAnkle"),
        "Húmero L":dist("LShoulder","LElbow"), "Húmero R":dist("RShoulder","RElbow"),
        "Antebrazo L":dist("LElbow","LWrist"), "Antebrazo R":dist("RElbow","RWrist"),
    }
    arr=df[["x","y","z"]].to_numpy(float)
    span=np.nanmax(arr,axis=0)-np.nanmin(arr,axis=0)
    st.markdown("**Control geométrico previo al fit**")
    q1,q2,q3=st.columns(3)
    q1.metric("Landmarks válidos",len(df))
    q2.metric("Extensión XYZ máx.",f"{float(np.max(span)):.3f}")
    finite=[v for v in segs.values() if np.isfinite(v) and v>0]
    q3.metric("Segmentos evaluables",len(finite))
    with st.expander("Longitudes del frame objetivo",expanded=False):
        st.dataframe(pd.DataFrame([{"segmento":k,"longitud_unidades_XYZ":v} for k,v in segs.items()]),use_container_width=True,hide_index=True)

    st.markdown("**Modelo SKEL privado · necesario para ejecutar el ajuste anatómico real**")
    st.caption("El puente XYZ ya está validado. Para que V110 genere y ajuste la malla esquelética SKEL real debes aportar tu ZIP oficial con `skel_male.pkl` o `skel_female.pkl`.")
    bundle=st.file_uploader("ZIP privado con los archivos de modelo SKEL descargados por ti desde el portal oficial",type=["zip"],key="v110_skel_private_bundle",help="No se guarda en Supabase ni se incorpora a la exportación de PhysioSentinel.")
    if bundle is None:
        st.info("Entrada articular preparada: 15 landmarks. Falta únicamente el modelo SKEL privado para ejecutar el ajuste anatómico real de este frame. V110 no sustituye SKEL por una malla falsa.")
        return
    raw=bundle.getvalue(); audit=_inspect_private_bundle(raw)
    if not audit.get("valid_zip"):
        st.error("El archivo aportado no es un ZIP SKEL válido."); return
    st.write({"skel_male.pkl":audit["has_male"],"skel_female.pkl":audit["has_female"]})
    if not (audit["has_male"] or audit["has_female"]):
        st.error("No encuentro skel_male.pkl ni skel_female.pkl en el ZIP. No se ejecuta ningún modelo."); return
    try:
        import torch  # noqa
        from skel.skel_model import SKEL  # type: ignore # noqa
        runtime=True
    except Exception:
        runtime=False
    if not runtime:
        st.warning("El bundle privado parece válido, pero el runtime Python `skel` no está instalado en esta instancia. V110 no realiza instalaciones dinámicas del runtime ni redistribuye SKEL. El modelo está presente, pero falta el paquete Python `skel` en esta instancia.")
    else:
        st.success("Runtime SKEL detectado. Runtime SKEL detectado. El siguiente paso dentro de V110 es ejecutar el fitting de q(46) contra estos 15 landmarks; no se extiende a 75 frames hasta validar este frame.")
