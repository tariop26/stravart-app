import streamlit as st
import cv2
import numpy as np
import folium
from streamlit_folium import st_folium
import xml.etree.ElementTree as ET
from xml.dom import minidom
from math import radians, cos, sin, asin, sqrt
import requests
import pandas as pd
import altair as alt
import urllib3
import osmnx as ox
import networkx as nx
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="StravArt Optimizer V9", layout="wide")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.requests_kwargs = {'verify': False}

# --- CORE FUNCTIONS ---

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r

@st.cache_resource
def load_graph(lat, lon, dist_km):
    try:
        # On charge large pour permettre le scan d'échelle
        custom_filter = '["highway"~"footway|path|track|service|residential|living_street|pedestrian|steps|unclassified|tertiary|secondary"]'
        G = ox.graph_from_point((lat, lon), dist=dist_km*1000, custom_filter=custom_filter, simplify=False)
        return G
    except Exception as e:
        return None

def get_elevations(locations):
    base_url = "https://api.open-meteo.com/v1/elevation"
    all_elevations = []
    chunk_size = 100
    for i in range(0, len(locations), chunk_size):
        chunk = locations[i:i + chunk_size]
        lats = [f"{lat:.4f}" for lat, lon in chunk]
        lons = [f"{lon:.4f}" for lat, lon in chunk]
        try:
            url = f"{base_url}?latitude={','.join(lats)}&longitude={','.join(lons)}"
            response = requests.get(url, verify=False, timeout=10)
            if response.status_code == 200:
                data = response.json()
                all_elevations.extend(data.get('elevation', [0.0]*len(chunk)))
            else:
                all_elevations.extend([0.0]*len(chunk))
        except:
            all_elevations.extend([0.0]*len(chunk))
    return all_elevations

def process_image_contours(image_file, invert=True):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image_file.seek(0)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, 0, 0
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray, 127, 255, thresh_type)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None, width, height
    normalized_segments = []
    for contour in contours:
        if cv2.contourArea(contour) < 50: continue 
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        seg = [(x/width, y/height) for x, y in points]
        if len(seg) > 2: seg.append(seg[0])
        normalized_segments.append(seg)
    return normalized_segments, width, height

def project_ghost(normalized_segments, start_lat, start_lon, target_dist_km, img_ratio):
    if not normalized_segments: return []
    base_dist = 0
    ref_lat_rad = radians(45)
    for seg in normalized_segments:
        for i in range(len(seg)-1):
            dx = (seg[i+1][0]-seg[i][0]) * 111.32 * cos(ref_lat_rad)
            dy = (seg[i+1][1]-seg[i][1]) * 111.32
            base_dist += sqrt(dx**2 + dy**2)
    scale = max(0.001, min(target_dist_km / base_dist, 0.5))
    scale_lat = scale
    scale_lon = scale * img_ratio * (1 / cos(radians(start_lat)))
    start_nx, start_ny = normalized_segments[0][0]
    theo_segs = []
    for seg in normalized_segments:
        gs = []
        for nx, ny in seg:
            lat = start_lat - ((ny - start_ny) * scale_lat) 
            lon = start_lon + ((nx - start_nx) * scale_lon)
            gs.append((lat, lon))
        theo_segs.append(gs)
    return theo_segs

def generate_gpx_xml(segments, creator_name):
    all_coords = [p for s in segments for p in s]
    elevs = get_elevations(all_coords)
    gpx = ET.Element("gpx", version="1.1", creator=creator_name, xmlns="http://www.topografix.com/GPX/1/1")
    trk = ET.SubElement(gpx, "trk")
    ET.SubElement(trk, "name").text = "Projet Strava Art"
    total_d, total_g = 0, 0
    chart_data = []
    idx, cumul = 0, 0
    for seg in segments:
        trkseg = ET.SubElement(trk, "trkseg")
        prev = None
        for lat, lon in seg:
            ele = elevs[idx] if idx < len(elevs) else 0
            idx += 1
            pt = ET.SubElement(trkseg, "trkpt", lat=f"{lat:.6f}", lon=f"{lon:.6f}")
            ET.SubElement(pt, "ele").text = f"{ele:.1f}"
            if prev:
                d = haversine(prev[1], prev[0], lon, lat)
                total_d += d
                cumul += d
                if ele > prev[2]: total_g += (ele - prev[2])
            chart_data.append({"Dist": cumul, "Alt": ele})
            prev = (lat, lon, ele)
    xml = minidom.parseString(ET.tostring(gpx)).toprettyxml(indent="  ")
    df = pd.DataFrame(chart_data)
    return xml, total_d, total_g, df

# --- LOGIQUE ROUTING ---
def calculate_distortion(theo_seg, real_seg):
    """Calcule l'erreur moyenne entre le tracé théorique et réel (en km)"""
    # Simplification : différence de longueur + distance des points extrêmes
    # Une métrique simple : Ratio de longueur. Si > 1, c'est un détour.
    len_theo = sum([haversine(theo_seg[i][1], theo_seg[i][0], theo_seg[i+1][1], theo_seg[i+1][0]) for i in range(len(theo_seg)-1)])
    len_real = sum([haversine(real_seg[i][1], real_seg[i][0], real_seg[i+1][1], real_seg[i+1][0]) for i in range(len(real_seg)-1)])
    if len_theo == 0: return 100
    return abs(len_real - len_theo)

def snap_hybrid_logic_optimized(G, segments_theoriques, tolerance):
    real_path_coords = []
    total_distortion = 0
    
    for segment in segments_theoriques:
        lats = [p[0] for p in segment]
        lons = [p[1] for p in segment]
        nearest_nodes = ox.distance.nearest_nodes(G, lons, lats)
        segment_coords = [segment[0]]

        for i in range(len(nearest_nodes) - 1):
            u, v = nearest_nodes[i], nearest_nodes[i+1]
            p1_theo, p2_theo = segment[i], segment[i+1]
            dist_direct = haversine(p1_theo[1], p1_theo[0], p2_theo[1], p2_theo[0])
            path_coords = []
            path_found = False
            
            if u != v:
                try:
                    # On utilise le poids 'length'
                    route = nx.shortest_path(G, u, v, weight='length')
                    
                    # Calcul longueur via G
                    dist_route = 0
                    temp = []
                    for k in range(len(route)-1):
                        u_n, v_n = route[k], route[k+1]
                        dist_route += G.get_edge_data(u_n, v_n)[0].get('length', 0) / 1000.0
                        node = G.nodes[v_n]
                        temp.append((node['y'], node['x']))
                    
                    if dist_route <= (dist_direct * tolerance) or dist_direct < 0.05:
                        path_coords = temp
                        path_found = True
                        total_distortion += abs(dist_route - dist_direct)
                    else:
                        # Penalty pour le mode hors-piste (on préfère quand ça suit la route)
                        # Mais ici on veut juste savoir si ça colle au dessin.
                        # Si on coupe tout droit, la distortion géométrique est faible (c'est le trait bleu)
                        # MAIS c'est infaisable. Donc on ajoute une pénalité virtuelle.
                        total_distortion += (dist_direct * 0.5) 
                        
                except: 
                    total_distortion += (dist_direct * 0.5) 
            
            if path_found: segment_coords.extend(path_coords)
            else: segment_coords.append(p2_theo)
        real_path_coords.append(segment_coords)
        
    return real_path_coords, total_distortion

def find_best_scale(G, norm_segs, start_lat, start_lon, img_ratio, min_dist, max_dist, step_dist):
    """Teste plusieurs échelles et trouve celle qui minimise la distorsion"""
    best_dist = min_dist
    min_error = float('inf')
    results = []
    
    scan_range = np.arange(min_dist, max_dist + 0.1, step_dist)
    
    progress_bar = st.progress(0, text="Scan des échelles...")
    
    for i, test_dist in enumerate(scan_range):
        progress_bar.progress(i / len(scan_range), text=f"Test échelle : {test_dist:.1f} km...")
        
        # 1. Générer Fantôme
        ghost = project_ghost(norm_segs, start_lat, start_lon, test_dist, img_ratio)
        
        # 2. Snap rapide (tolérance stricte 1.3 pour pénaliser les détours)
        _, error = snap_hybrid_logic_optimized(G, ghost, tolerance=1.3)
        
        # Normaliser l'erreur par la distance totale (pour ne pas avantager les petits dessins)
        normalized_error = error / test_dist
        
        results.append((test_dist, normalized_error))
        
        if normalized_error < min_error:
            min_error = normalized_error
            best_dist = test_dist
            
    progress_bar.empty()
    return best_dist, results

def calculate_manual_segment(G, start_coords, end_coords, offroad):
    if offroad or G is None: return [start_coords, end_coords]
    try:
        u = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
        v = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
        path = nx.shortest_path(G, u, v, weight='length')
        return [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
    except: return [start_coords, end_coords]

# --- SESSION STATE ---
if 'auto_start' not in st.session_state: st.session_state['auto_start'] = [45.3533, 5.6254]
if 'auto_results' not in st.session_state: st.session_state['auto_results'] = None
if 'manual_waypoints' not in st.session_state: st.session_state['manual_waypoints'] = []
if 'manual_trace' not in st.session_state: st.session_state['manual_trace'] = []
if 'manual_ghost' not in st.session_state: st.session_state['manual_ghost'] = None
if 'graph_manual' not in st.session_state: st.session_state['graph_manual'] = None
# Pour l'optimiseur
if 'best_scale_found' not in st.session_state: st.session_state['best_scale_found'] = None

# --- INTERFACE ---
st.title("🎨 StravArt Ultimate Suite V9")

tab1, tab2 = st.tabs(["🚀 Mode Auto & Optimiseur", "✍️ Mode Manuel"])

# ==========================================
# ONGLET 1 : AUTOMATIQUE + OPTIMISEUR
# ==========================================
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Paramètres")
        auto_img = st.file_uploader("Image", type=['jpg', 'png'], key="auto_up")
        
        # Si on a trouvé une meilleure échelle, on met à jour la valeur par défaut
        default_dist = 10.0
        if st.session_state['best_scale_found']:
            default_dist = st.session_state['best_scale_found']
            
        auto_dist = st.slider("Distance Cible (km)", 2.0, 30.0, default_dist, 0.5, key="auto_dist")
        
        st.markdown("---")
        st.markdown("##### 🧠 Intelligence Artificielle")
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            if st.button("🪄 Trouver la meilleure taille", disabled=not auto_img, help="Teste plusieurs tailles et trouve celle qui colle le mieux aux routes existantes."):
                with st.spinner("Analyse topologique en cours..."):
                    norm_segs, w, h = process_image_contours(auto_img)
                    if norm_segs:
                        # On charge le graphe une fois pour une zone large (30km max)
                        G = load_graph(st.session_state['auto_start'][0], st.session_state['auto_start'][1], 15.0)
                        if G:
                            best_d, res = find_best_scale(
                                G, norm_segs, 
                                st.session_state['auto_start'][0], 
                                st.session_state['auto_start'][1], 
                                w/h, 
                                min_dist=4.0, max_dist=20.0, step_dist=2.0
                            )
                            st.session_state['best_scale_found'] = best_d
                            st.success(f"Taille optimale trouvée : {best_d:.1f} km !")
                            st.rerun()
                        else:
                            st.error("Impossible de charger la carte pour l'analyse.")

        with col_opt2:
            snap_on = st.checkbox("Snap-to-Road", value=True)

        if st.button("Calculer Itinéraire", type="primary", disabled=not auto_img):
            with st.spinner("Génération..."):
                norm_segs, w, h = process_image_contours(auto_img)
                if norm_segs:
                    # Projection
                    ghost = project_ghost(norm_segs, st.session_state['auto_start'][0], st.session_state['auto_start'][1], auto_dist, w/h)
                    
                    # Routing
                    final_segs = ghost
                    if snap_on:
                        G = load_graph(st.session_state['auto_start'][0], st.session_state['auto_start'][1], auto_dist/2)
                        if G: final_segs, _ = snap_hybrid_logic_optimized(G, ghost, tolerance=1.5)
                    
                    # Export
                    xml, d, g, df = generate_gpx_xml(final_segs, "StravArt_Auto")
                    st.session_state['auto_results'] = {'real': final_segs, 'ghost': ghost, 'xml': xml, 'd': d, 'g': g, 'df': df}
    
        if st.session_state['auto_results']:
            res = st.session_state['auto_results']
            st.divider()
            st.metric("Distance", f"{res['d']:.2f} km")
            st.metric("D+ Estimé", f"{int(res['g'])} m")
            st.download_button("Télécharger GPX", res['xml'], "stravart_auto.gpx")

    with c2:
        m_auto = folium.Map(location=st.session_state['auto_start'], zoom_start=13)
        folium.Marker(st.session_state['auto_start'], icon=folium.Icon(color="green", icon="play"), popup="Départ").add_to(m_auto)
        
        if st.session_state['auto_results']:
            for s in st.session_state['auto_results']['ghost']:
                folium.PolyLine(s, color="blue", weight=2, opacity=0.4, dash_array='5,5').add_to(m_auto)
            for s in st.session_state['auto_results']['real']:
                folium.PolyLine(s, color="red", weight=4, opacity=0.8).add_to(m_auto)
            
            c = alt.Chart(st.session_state['auto_results']['df']).mark_area(color='red', opacity=0.3).encode(x='Dist', y='Alt').properties(height=150)
            st.altair_chart(c, use_container_width=True)

        out_auto = st_folium(m_auto, width=800, height=600, key="map_auto")
        if out_auto and out_auto['last_clicked']:
             if abs(out_auto['last_clicked']['lat'] - st.session_state['auto_start'][0]) > 0.0001:
                st.session_state['auto_start'] = [out_auto['last_clicked']['lat'], out_auto['last_clicked']['lng']]
                st.rerun()

# ==========================================
# ONGLET 2 : MANUEL (CODE IDENTIQUE PRECEDENT)
# ==========================================
with tab2:
    cm1, cm2 = st.columns([1, 3])
    with cm1:
        st.subheader("Traceur Manuel")
        man_img = st.file_uploader("Image Modèle", type=['jpg', 'png'], key="man_up")
        start_lat_m = st.number_input("Lat Départ", value=st.session_state['auto_start'][0], format="%.5f", key="slm")
        start_lon_m = st.number_input("Lon Départ", value=st.session_state['auto_start'][1], format="%.5f", key="slom")
        man_dist = st.slider("Taille Modèle (km)", 2.0, 30.0, 10.0, key="man_dist")
        
        if st.button("📍 Poser le Fantôme", type="primary"):
            if man_img:
                norm_segs, w, h = process_image_contours(man_img)
                if norm_segs:
                    st.session_state['manual_ghost'] = project_ghost(norm_segs, start_lat_m, start_lon_m, man_dist, w/h)
                    st.session_state['manual_waypoints'] = []
                    st.session_state['manual_trace'] = []
                    with st.spinner("Chargement carte..."):
                        st.session_state['graph_manual'] = load_graph(start_lat_m, start_lon_m, man_dist/2)
                    st.rerun()

        st.markdown("---")
        man_offroad = st.checkbox("🥾 Mode Hors-Piste", value=False, key="moff")
        
        if st.button("↩️ Undo Dernier Point"):
            if len(st.session_state['manual_waypoints']) > 0:
                st.session_state['manual_waypoints'].pop()
                st.session_state['manual_trace'] = [] # Reset visuel simple
                st.rerun()
                
        if len(st.session_state['manual_trace']) > 1:
            xml_m, d_m, g_m, _ = generate_gpx_xml([st.session_state['manual_trace']], "StravArt_Manual")
            st.metric("Distance", f"{d_m:.2f} km")
            st.download_button("💾 Sauvegarder GPX", xml_m, "mon_trace_manuel.gpx")

    with cm2:
        center_m = [start_lat_m, start_lon_m]
        if st.session_state['manual_waypoints']: center_m = st.session_state['manual_waypoints'][-1]
        m_man = folium.Map(location=center_m, zoom_start=15)
        
        if st.session_state['manual_ghost']:
            for s in st.session_state['manual_ghost']:
                folium.PolyLine(s, color="blue", weight=3, opacity=0.4, dash_array='10,10').add_to(m_man)
        if st.session_state['manual_trace']:
            folium.PolyLine(st.session_state['manual_trace'], color="red", weight=5, opacity=0.8).add_to(m_man)
        for wp in st.session_state['manual_waypoints']:
             folium.CircleMarker(wp, radius=4, color="black", fill=True, fill_color="white").add_to(m_man)
             
        out_man = st_folium(m_man, width=900, height=700, key="map_man")
        if out_man and out_man['last_clicked']:
            new_pt = (out_man['last_clicked']['lat'], out_man['last_clicked']['lng'])
            if not st.session_state['manual_waypoints'] or st.session_state['manual_waypoints'][-1] != new_pt:
                if not st.session_state['manual_waypoints']:
                    st.session_state['manual_waypoints'].append(new_pt)
                    st.session_state['manual_trace'].append(new_pt)
                else:
                    last_pt = st.session_state['manual_waypoints'][-1]
                    seg = calculate_manual_segment(st.session_state['graph_manual'], last_pt, new_pt, man_offroad)
                    st.session_state['manual_waypoints'].append(new_pt)
                    st.session_state['manual_trace'].extend(seg[1:])
                st.rerun()